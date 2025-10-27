#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_bisection_S3_cached_vm.py
Versione *cache-first* (NO OVERWRITE se risultato (c, Δ) esiste già):
- Ogni combinazione (c, Δ_trial) scrive in una cartella distinta runs/c_<c>/d_<Δ>.
- Se la cartella contiene un risultato valido (iterations.json/out.txt), il run viene SALTATO.
- Nessun hard reset globale: i run precedenti NON vengono cancellati.
- Retry SDPB opzionale come nel logger_vm (rigenera .sdp dentro la cartella locale).
"""

import json, os, subprocess, time, shutil, logging
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# Radice di lavoro e file/script condivisi
# ──────────────────────────────────────────────────────────────────────────────
WORKDIR       = Path("/home/azureuser/bootstrap_data").resolve()
RUNS_ROOT     = WORKDIR / "runs"              # tutte le cartelle per (c,Δ)
WLS_SCRIPT    = WORKDIR / "PROBLEM_to_JSON_s3_script.wls"
CONVERTER     = WORKDIR / "JSON_to_JSONCLEAN_OK0.py"

LOG_FILE      = WORKDIR / "run_bisection_cached.log"
RESULTS_JSON  = WORKDIR / "bound_results_S3_cached.json"

# Docker/SDPB
DOCKER_IMAGE  = "bootstrapcollaboration/sdpb:master"
DOCKER_WORK   = "/work"

# ──────────────────────────────────────────────────────────────────────────────
# Parametri via environment (con default sensati)
# ──────────────────────────────────────────────────────────────────────────────
NMAX      = int(os.environ.get("NMAX", 4))
C_LIST    = [float(x) for x in os.environ.get("C_LIST", "1,2,3,4").split(",")]
DELTA_LO  = float(os.environ.get("DELTA_LO", 0.0))
DELTA_HI  = float(os.environ.get("DELTA_HI", 2.0))
TOL       = float(os.environ.get("TOL", 2e-3))

# Precisioni / soglie
PMP2SDP_PREC = int(os.environ.get("PMP2SDP_PREC", 2048))
SDPB_PROFILE = dict(
    precision = int(os.environ.get("SDPB_PREC", 2048)),
    maxIters  = int(os.environ.get("SDPB_MAXITERS", 1800)),
    gap       = os.environ.get("SDPB_GAP", "1e-45"),
    primalErr = os.environ.get("SDPB_PRIM_ERR", "1e-35"),
    dualErr   = os.environ.get("SDPB_DUAL_ERR", "1e-35"),
    scalePr   = int(os.environ.get("SDPB_SCALE_PR", 8)),
    scaleDu   = int(os.environ.get("SDPB_SCALE_DU", 8)),
    verbosity = os.environ.get("SDPB_VERBOSITY", "2"),
)
SDPB_STEP       = os.environ.get("SDPB_STEP", None)
SDPB_NP         = int(os.environ.get("SDPB_NP", "1"))
NUMERIC_AS_DUAL = bool(int(os.environ.get("NUMERIC_AS_DUAL", "0")))

# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(LOG_FILE, mode="a")]
)
log = logging.getLogger()
log.info("Profilo SDPB: precision=%s, maxIters=%s, gap=%s, primErr=%s, dualErr=%s, "
         "scalePr=%s, scaleDu=%s, verb=%s, step=%s",
         SDPB_PROFILE['precision'], SDPB_PROFILE['maxIters'], SDPB_PROFILE['gap'],
         SDPB_PROFILE['primalErr'], SDPB_PROFILE['dualErr'],
         SDPB_PROFILE['scalePr'], SDPB_PROFILE['scaleDu'],
         SDPB_PROFILE['verbosity'], SDPB_STEP)

# ──────────────────────────────────────────────────────────────────────────────
# Utility
# ──────────────────────────────────────────────────────────────────────────────
def ftag(x: float) -> str:
    """Tag dei float in filename: 1.25 → 1p25 ; 0.003 → 0p003"""
    s = f"{x:.12g}"
    return s.replace('.', 'p').replace('-', 'm')

def run_logged(cmd, cwd=None, env=None) -> int:
    cmd_str = " ".join(cmd)
    log.info("ESEGUO: %s", cmd_str)
    try:
        p = subprocess.run(cmd, cwd=cwd, env=env)
        rc = p.returncode
        if rc != 0:
            log.error("Comando fallito (rc=%d): %s", rc, cmd_str)
        return rc
    except FileNotFoundError:
        log.error("Comando non trovato: %s", cmd_str)
        return 127
    except Exception:
        log.exception("Eccezione eseguendo: %s", cmd_str)
        return 1

def time_block(fn, *args, **kwargs):
    t0 = time.time()
    rc = fn(*args, **kwargs)
    return rc, time.time() - t0

# ──────────────────────────────────────────────────────────────────────────────
# Parsing esiti SDPB (versione robusta a JSON troncati)
# ──────────────────────────────────────────────────────────────────────────────
def parse_status(run_dir: Path) -> str:
    """Ritorna 'PRIMAL' / 'DUAL' / 'UNKNOWN' leggendo iterations.json o out.txt.
    Robusto a file JSON troncati/incompleti.
    """
    it_json = run_dir / "sdpb_outS3" / "iterations.json"
    out_txt = run_dir / "sdpb_outS3" / "out.txt"

    # 1) iterations.json (tollerante)
    if it_json.exists():
        raw = it_json.read_text(errors="ignore").strip()
        if raw:
            try:
                # parsing completo
                it = json.loads(raw)
            except json.JSONDecodeError:
                # prova a troncare al pattern '}]' (fine lista di oggetti)
                last_closing = raw.rfind("}]")
                if last_closing != -1:
                    try:
                        it = json.loads(raw[:last_closing + 2])
                    except Exception:
                        it = []
                else:
                    it = []
            except Exception:
                it = []

            if isinstance(it, list) and it:
                last = it[-1]
                pf = bool(last.get("primalFeasible", False))
                df = bool(last.get("dualFeasible", False))
                if pf and not df:
                    return "PRIMAL"
                if df and not pf:
                    return "DUAL"
                if pf and df:
                    return "PRIMAL"

    # 2) out.txt (fallback)
    if out_txt.exists():
        try:
            txt = out_txt.read_text(errors="ignore")
            if "found primal-dual optimal solution" in txt:
                return "PRIMAL"
            if "found primal feasible solution" in txt:
                return "PRIMAL"
            if "found dual feasible solution" in txt:
                return "DUAL"
            if "primalFeasible: true" in txt and "dualFeasible: true" not in txt:
                return "PRIMAL"
            if "dualFeasible: true" in txt and "primalFeasible: true" not in txt:
                return "DUAL"
        except Exception:
            pass

    return "UNKNOWN"

# ──────────────────────────────────────────────────────────────────────────────
# Pipeline per un singolo (c, Δ_trial) dentro la sua cartella
# ──────────────────────────────────────────────────────────────────────────────
def write_params_for_nb(run_dir: Path, c: float, delta_trial: float):
    data = {"c": c, "DeltaTrial": float(delta_trial), "N_max": NMAX}
    (run_dir / "params_for_nb_S3.json").write_text(json.dumps(data))

def run_wolfram(run_dir: Path):
    cmd = ["wolframscript", "-file", str(WLS_SCRIPT)]
    # usiamo cwd=run_dir così i file escono dentro
    return time_block(run_logged, cmd, run_dir)

def run_cleaner_and_pmp2sdp(run_dir: Path, pmp_prec=None) -> Path:
    """Pulizia JSON e conversione in .sdp nella cartella del run."""
    raw_json   = run_dir / "out_raw_S3.json"
    clean_json = run_dir / "fileS3_clean.json"
    sdp_file   = run_dir / "fileS3_clean.sdp"
    sdpb_out   = run_dir / "sdpb_outS3"

    log.info("Pulizia JSON → %s …", clean_json.name)
    if not CONVERTER.exists():
        log.error("Script converter mancante: %s", CONVERTER)
        raise FileNotFoundError(str(CONVERTER))

    # 1) pulisci target JSON
    try: clean_json.unlink()
    except FileNotFoundError: pass

    rc, _dt = time_block(run_logged, ["python3", str(CONVERTER), str(raw_json), str(clean_json)])
    if rc != 0 or not clean_json.exists():
        raise RuntimeError("Pulizia JSON fallita")

    # 2) elimina eventuale .sdp precedente nella cartella del run
    try:
        if sdp_file.exists():
            if sdp_file.is_dir():
                shutil.rmtree(sdp_file, ignore_errors=True)
            else:
                sdp_file.unlink()
    except FileNotFoundError:
        pass

    # 3) pmp2sdp via Docker (precisione coerente)
    use_prec = int(pmp_prec) if pmp_prec is not None else PMP2SDP_PREC
    log.info("pmp2sdp → %s …", sdp_file.name)
    docker_cmd = [
        "docker","run","--rm",
        "-v", f"{run_dir}:{DOCKER_WORK}",
        DOCKER_IMAGE,
        "pmp2sdp",
        "--input",  f"{DOCKER_WORK}/{clean_json.name}",
        "--output", f"{DOCKER_WORK}/{sdp_file.name}",
        "--precision", str(use_prec),
    ]
    rc, _dt = time_block(run_logged, docker_cmd)
    if rc != 0 or not sdp_file.exists():
        raise RuntimeError("pmp2sdp (Docker) fallito o output .sdp mancante")
    return sdp_file

def _sdpb_cmd_base_args(sdp_path: Path, run_dir: Path):
    outdir = run_dir / "sdpb_outS3"
    base = [
        "--precision", str(SDPB_PROFILE['precision']),
        "--maxIterations", str(SDPB_PROFILE['maxIters']),
        "--dualityGapThreshold", SDPB_PROFILE['gap'],
        "--primalErrorThreshold", SDPB_PROFILE['primalErr'],
        "--dualErrorThreshold", SDPB_PROFILE['dualErr'],
        "--initialMatrixScalePrimal", str(SDPB_PROFILE['scalePr']),
        "--initialMatrixScaleDual",   str(SDPB_PROFILE['scaleDu']),
        "--verbosity", SDPB_PROFILE['verbosity'],
        "--findPrimalFeasible",
        "--findDualFeasible",
        "--detectPrimalFeasibleJump",
        "--detectDualFeasibleJump",
        "--noFinalCheckpoint",
        "-s", f"{DOCKER_WORK}/{sdp_path.name}",
        "-o", f"{DOCKER_WORK}/{outdir.name}",
    ]
    if SDPB_STEP is not None:
        base = ["--stepLengthReduction", SDPB_STEP] + base
    return base

def run_sdpb_once(sdp_path: Path, run_dir: Path) -> dict:
    # ricrea cartella output dentro il run_dir
    outdir = run_dir / "sdpb_outS3"
    if outdir.exists():
        shutil.rmtree(outdir, ignore_errors=True)
    outdir.mkdir(parents=True, exist_ok=True)

    base = _sdpb_cmd_base_args(sdp_path, run_dir)
    inner = (["mpirun","-np",str(SDPB_NP),"sdpb"] + base) if SDPB_NP > 1 else (["sdpb"] + base)
    docker_cmd = ["docker","run","--rm","-v", f"{run_dir}:{DOCKER_WORK}", DOCKER_IMAGE] + inner
    rc, _dt = time_block(run_logged, docker_cmd)

    status = parse_status(run_dir)
    if rc != 0:
        log.error("SDPB rc=%d (status parse: %s)", rc, status)
    return {"status": status, "rc": rc}

def run_sdpb_with_retry(run_dir: Path, sdp_path: Path) -> dict:
    """Retry con precisione↑ e scale 1/1, rigenerando lo .sdp *nella cartella locale*."""
    r = run_sdpb_once(sdp_path, run_dir)
    if r["rc"] == 0:
        return r

    log.warning("SDPB fallito: retry con precisione↑ e scaling 1/1 + rigenera .sdp")
    old_prec, old_pr, old_du = SDPB_PROFILE['precision'], SDPB_PROFILE['scalePr'], SDPB_PROFILE['scaleDu']
    try:
        SDPB_PROFILE['precision'] = old_prec + 512
        SDPB_PROFILE['scalePr'] = 1
        SDPB_PROFILE['scaleDu'] = 1

        # rigenera lo .sdp alla NUOVA precisione
        try:
            sdp_path = run_cleaner_and_pmp2sdp(run_dir, pmp_prec=SDPB_PROFILE['precision'])
        except Exception as e:
            log.error("Rigenerazione .sdp al retry fallita: %s", e)
            return {"status": "UNKNOWN", "rc": 1}

        return run_sdpb_once(sdp_path, run_dir)
    finally:
        SDPB_PROFILE['precision'] = old_prec
        SDPB_PROFILE['scalePr']   = old_pr
        SDPB_PROFILE['scaleDu']   = old_du

# ──────────────────────────────────────────────────────────────────────────────
# Orchestratore singolo Δ_trial con CACHE CHECK
# ──────────────────────────────────────────────────────────────────────────────
def ensure_run_dir(c: float, delta_trial: float) -> Path:
    ctag = f"c_{ftag(c)}"
    dtag = f"d_{ftag(delta_trial)}"
    rd = RUNS_ROOT / ctag / dtag
    rd.mkdir(parents=True, exist_ok=True)
    return rd

def cached_status(run_dir: Path) -> str:
    """Se esiste già un risultato, restituiscilo e salta il run."""
    status = parse_status(run_dir)
    if status != "UNKNOWN":
        log.info("CACHE HIT in %s → %s", run_dir.name, status)
        return status
    return "UNKNOWN"

def check_once(c: float, delta_trial: float) -> str:
    run_dir = ensure_run_dir(c, delta_trial)
    log.info("c=%g | Δ_trial=%g | N_max=%d | run_dir=%s", c, delta_trial, NMAX, run_dir)

    # 0) CACHE: se abbiamo già out valido, non sovrascrivere
    st = cached_status(run_dir)
    if st != "UNKNOWN":
        return st

    # 1) scrivi params e lancia Wolfram DENTRO run_dir
    write_params_for_nb(run_dir, c, delta_trial)
    log.info("Wolfram → genera out_raw_S3.json (in %s) …", run_dir.name)
    rc, dt = run_wolfram(run_dir)
    if rc != 0:
        log.error("wolframscript fallito (rc=%d).", rc)
        return "UNKNOWN"
    log.info("OK (%.2fs)", dt)

    # 2) Cleaner + pmp2sdp
    try:
        sdp_path = run_cleaner_and_pmp2sdp(run_dir, pmp_prec=PMP2SDP_PREC)
    except Exception as e:
        log.error("Cleaner/pmp2sdp errore: %s", e)
        return "UNKNOWN"

    # 3) SDPB con retry
    log.info("SDPB → solve …")
    r = run_sdpb_with_retry(run_dir, sdp_path)
    status = r.get("status", "UNKNOWN")

    if status == "UNKNOWN" and NUMERIC_AS_DUAL:
        log.warning("Status UNKNOWN; NUMERIC_AS_DUAL=1 → tratto come DUAL per separazione.")
        status = "DUAL"

    log.info("Stato SDPB: %s", status)
    return status

# ──────────────────────────────────────────────────────────────────────────────
# Separazione e bisezione (immutate nella logica)
# ──────────────────────────────────────────────────────────────────────────────
def separate_interval(c: float, lo: float, hi: float):
    slo = check_once(c, lo)
    shi = check_once(c, hi)

    tries_up = 0
    while slo == shi and tries_up < 3:
        hi *= 1.5
        log.warning("Intervallo non separato (slo=shi=%s). Estendo hi → %.3f", slo, hi)
        shi = check_once(c, hi)
        tries_up += 1

    tries_down = 0
    while slo == shi and tries_down < 3 and lo > 1e-6:
        lo = max(1e-6, lo/2)
        log.warning("Ancora non separato. Riduco lo → %.6f", lo)
        slo = check_once(c, lo)
        tries_down += 1

    if slo == shi:
        raise RuntimeError(f"[c={c}] impossibile inizializzare bisezione (stesso stato a lo/hi).")

    return lo, slo, hi, shi

def bisect(c: float, lo: float, slo: str, hi: float, shi: str) -> float:
    while (hi - lo) > TOL:
        mid = 0.5 * (lo + hi)
        smid = check_once(c, mid)
        if smid == slo:
            lo, slo = mid, smid
        elif smid == shi:
            hi, shi = mid, smid
        else:
            # UNKNOWN → dilata leggermente per uscire dalla zona instabile
            hi *= 1.1
            lo = max(1e-6, lo / 1.1)
    return 0.5 * (lo + hi)

# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    log.info("=== Avvio scansione (Nmax=%d) [cache-first] ===", NMAX)
    log.info("Log dettagliato: %s", LOG_FILE)
    log.info("Cartella runs: %s", RUNS_ROOT)
    log.info("────────────────────────────────────────────────────────")

    # Riduci contesa BLAS (utile nei container)
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    results = []
    for c in C_LIST:
        log.info("Bisezione su c=%g", c)
        log.warning("BISEZIONE c=%g in [%.3f, %.3f] (tol=%.3g) con N_max=%d",
                    c, DELTA_LO, DELTA_HI, TOL, NMAX)
        try:
            lo, slo, hi, shi = separate_interval(c, DELTA_LO, DELTA_HI)
            bound = bisect(c, lo, slo, hi, shi)
            log.info("[c=%.6g] Bound ≈ %.6f", c, bound)
            results.append({"c": c, "bound": bound, "Nmax": NMAX})
        except Exception as e:
            log.error("%s", e)
            log.warning("Nessun risultato.")

    RESULTS_JSON.write_text(json.dumps(results, indent=2))
    log.info("Salvato: %s", RESULTS_JSON)
    log.info("Fine.")
