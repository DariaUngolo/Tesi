#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_bisection_commentato_vm_FIXED2.py

Versione robusta e "cache-first" ispirata a run_bisection_S3_loggger_vm1.py:
- Ogni (c, Δ) vive in runs/c_<c>/d_<Δ> e NON viene sovrascritto se già valido.
- Parsing robusto di iterations.json/out.txt (SDPB 3.x).
- Bisezione con separazione automatica dell’intervallo (estensione hi/riduzione lo).
- MPI inside Docker: di default abilita --use-hwthread-cpus per evitare l'errore "not enough slots".
- Evita bound=0: la bisezione procede SOLO dopo separazione DUAL/PRIMAL.
"""

import json, os, subprocess, time, shutil, logging, re
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# Radici e script
# ──────────────────────────────────────────────────────────────────────────────
WORKDIR      = Path("/home/azureuser/bootstrap_data").resolve()
RUNS_ROOT    = WORKDIR / "runs"
# Di default uso lo script originale "vm"; cambia con env WLS_SCRIPT se necessario
WLS_SCRIPT   = Path(os.environ.get("WLS_SCRIPT", str(WORKDIR / "PROBLEM_to_JSON_script_vm.wls"))).resolve()
CONVERTER    = Path(os.environ.get("CONVERTER",   str(WORKDIR / "JSON_to_JSONCLEAN_OK0.py"))).resolve()

LOG_FILE     = WORKDIR / "run_bisection_vm_fixed2.log"
RESULTS_JSON = WORKDIR / "bound_results.json"

# Nomi dei file che ci aspettiamo dal notebook/Mathematica
RAW_JSON_NAME   = os.environ.get("RAW_JSON_NAME",   "out_raw.json")
CLEAN_JSON_NAME = os.environ.get("CLEAN_JSON_NAME", "file_clean.json")
SDP_NAME        = os.environ.get("SDP_NAME",        "file_clean.sdp")
SDPB_OUT_DIR    = os.environ.get("SDPB_OUT_DIR",    "sdpb_out")

# Docker/SDPB
DOCKER_IMAGE    = os.environ.get("DOCKER_IMAGE", "bootstrapcollaboration/sdpb:master")
DOCKER_WORK     = "/work"

# ──────────────────────────────────────────────────────────────────────────────
# Parametri bisezione e SDPB configurabili via env
# ──────────────────────────────────────────────────────────────────────────────
NMAX      = int(os.environ.get("NMAX", 9))
C_LIST    = [float(x) for x in os.environ.get("C_LIST", "0.8,1.0,1.5,2.0").split(",")]
DELTA_LO  = float(os.environ.get("DELTA_LO", 0.0))
DELTA_HI  = float(os.environ.get("DELTA_HI", 18.0))
TOL       = float(os.environ.get("TOL", 2e-3))

PMP2SDP_PREC = int(os.environ.get("PMP2SDP_PREC", 2048))

SDPB_PREC     = int(os.environ.get("SDPB_PREC", 2048))
SDPB_MAXITERS = int(os.environ.get("SDPB_MAXITERS", 1800))
SDPB_GAP      = os.environ.get("SDPB_GAP", "1e-60")
SDPB_PERR     = os.environ.get("SDPB_PRIM_ERR", "1e-40")
SDPB_DERR     = os.environ.get("SDPB_DUAL_ERR", "1e-40")
SDPB_SCALE_PR = int(os.environ.get("SDPB_SCALE_PR", 10))
SDPB_SCALE_DU = int(os.environ.get("SDPB_SCALE_DU", 10))
SDPB_VERB     = os.environ.get("SDPB_VERBOSITY", "1")
SDPB_STEP     = os.environ.get("SDPB_STEP", "0.3")
SDPB_NP       = int(os.environ.get("SDPB_NP", "4"))  # processi MPI
USE_HWTHREADS = int(os.environ.get("SDPB_USE_HWTHREAD_CPUS", "1"))  # 1 = abilita --use-hwthread-cpus

# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(LOG_FILE, mode="a")]
)
log = logging.getLogger()
log.info("WORKDIR=%s | RUNS=%s", WORKDIR, RUNS_ROOT)
log.info("SDPB: prec=%d iters=%d gap=%s Perr=%s Derr=%s scalePr=%d scaleDu=%d verb=%s step=%s np=%d hwthr=%d",
         SDPB_PREC, SDPB_MAXITERS, SDPB_GAP, SDPB_PERR, SDPB_DERR,
         SDPB_SCALE_PR, SDPB_SCALE_DU, SDPB_VERB, SDPB_STEP, SDPB_NP, USE_HWTHREADS)

# ──────────────────────────────────────────────────────────────────────────────
# Utils
# ──────────────────────────────────────────────────────────────────────────────
def ftag(x: float) -> str:
    return f"{x:.12g}".replace('.', 'p').replace('-', 'm')

def run_logged(cmd, cwd=None, env=None) -> int:
    log.info(">> %s", " ".join(cmd))
    try:
        p = subprocess.run(cmd, cwd=cwd, env=env)
        rc = p.returncode
        if rc != 0:
            log.error("Comando fallito (rc=%d): %s", rc, " ".join(cmd))
        return rc
    except Exception:
        log.exception("Eccezione eseguendo comando")
        return 1

def time_block(fn, *args, **kwargs):
    t0 = time.time()
    rc = fn(*args, **kwargs)
    return rc, time.time() - t0

# ──────────────────────────────────────────────────────────────────────────────
# Parsing esito SDPB (robusto a JSON troncati)
# ──────────────────────────────────────────────────────────────────────────────
def parse_status(run_dir: Path) -> str:
    it_json = run_dir / SDPB_OUT_DIR / "iterations.json"
    out_txt = run_dir / SDPB_OUT_DIR / "out.txt"

    if it_json.exists():
        raw = it_json.read_text(errors="ignore").strip()
        if raw:
            try:
                it = json.loads(raw)
            except json.JSONDecodeError:
                last = raw.rfind("}]")
                try:
                    it = json.loads(raw[:last+2]) if last != -1 else []
                except Exception:
                    it = []
            except Exception:
                it = []

            if isinstance(it, list) and it:
                last = it[-1]
                pf = bool(last.get("primalFeasible", False))
                df = bool(last.get("dualFeasible", False))
                if pf and not df: return "PRIMAL"
                if df and not pf: return "DUAL"
                if pf and df:     return "PRIMAL"

    if out_txt.exists():
        txt = out_txt.read_text(errors="ignore")
        if re.search(r"found primal-dual optimal solution", txt): return "PRIMAL"
        if re.search(r"found primal feasible solution", txt):     return "PRIMAL"
        if re.search(r"found dual feasible solution", txt):       return "DUAL"
        if "primalFeasible: true" in txt and "dualFeasible: true" not in txt: return "PRIMAL"
        if "dualFeasible: true" in txt and "primalFeasible: true" not in txt:  return "DUAL"

    return "UNKNOWN"

# ──────────────────────────────────────────────────────────────────────────────
# Pipeline per un singolo (c, Δ) nella sua cartella
# ──────────────────────────────────────────────────────────────────────────────
def ensure_run_dir(c: float, delta_trial: float) -> Path:
    rd = RUNS_ROOT / f"c_{ftag(c)}" / f"d_{ftag(delta_trial)}"
    rd.mkdir(parents=True, exist_ok=True)
    return rd

def write_params_for_nb(run_dir: Path, c: float, delta_trial: float):
    (run_dir / "params_for_nb.json").write_text(json.dumps({"c": c, "DeltaTrial": float(delta_trial), "N_max": NMAX}))

def run_wolfram(run_dir: Path) -> int:
    # Eseguo in cwd=run_dir per avere file locali; se lo script scrive in WORKDIR li copieremo dopo.
    return run_logged(["wolframscript", "-file", str(WLS_SCRIPT)], cwd=run_dir)

def ensure_raw_json(run_dir: Path) -> Path:
    """Garantisce che esista RAW_JSON_NAME dentro run_dir, copiando da WORKDIR se necessario."""
    local_raw = run_dir / RAW_JSON_NAME
    if local_raw.exists():
        return local_raw
    fallback = WORKDIR / RAW_JSON_NAME
    if fallback.exists():
        shutil.copy2(fallback, local_raw)
        return local_raw
    raise FileNotFoundError(f"Raw JSON non trovato né in {local_raw} né in {fallback}")

def run_cleaner_and_pmp2sdp(run_dir: Path, pmp_prec: int) -> Path:
    raw_json   = ensure_raw_json(run_dir)
    clean_json = run_dir / CLEAN_JSON_NAME
    sdp_file   = run_dir / SDP_NAME

    # Pulizia
    try:
        if clean_json.exists(): clean_json.unlink()
    except Exception: pass
    rc = run_logged(["python3", str(CONVERTER), str(raw_json), str(clean_json)])
    if rc != 0 or not clean_json.exists():
        raise RuntimeError("Pulizia JSON fallita")

    # Rigenera .sdp
    try:
        if sdp_file.exists():
            if sdp_file.is_dir(): shutil.rmtree(sdp_file, ignore_errors=True)
            else: sdp_file.unlink()
    except Exception: pass

    docker_cmd = [
        "docker","run","--rm",
        "-v", f"{run_dir}:{DOCKER_WORK}",
        DOCKER_IMAGE,
        "pmp2sdp",
        "--input",  f"{DOCKER_WORK}/{clean_json.name}",
        "--output", f"{DOCKER_WORK}/{sdp_file.name}",
        "--precision", str(int(pmp_prec)),
    ]
    rc, _dt = time_block(run_logged, docker_cmd)
    if rc != 0 or not sdp_file.exists():
        raise RuntimeError("pmp2sdp fallito o .sdp mancante")
    return sdp_file

def sdpb_base_args(sdp_path: Path, run_dir: Path):
    args = [
        "--precision", str(SDPB_PREC),
        "--maxIterations", str(SDPB_MAXITERS),
        "--dualityGapThreshold", SDPB_GAP,
        "--primalErrorThreshold", SDPB_PERR,
        "--dualErrorThreshold", SDPB_DERR,
        "--initialMatrixScalePrimal", str(SDPB_SCALE_PR),
        "--initialMatrixScaleDual",   str(SDPB_SCALE_DU),
        "--verbosity", SDPB_VERB,
        "--findDualFeasible",
        "--noFinalCheckpoint",
        "-s", f"{DOCKER_WORK}/{sdp_path.name}",
        "-o", f"{DOCKER_WORK}/{SDPB_OUT_DIR}",
    ]
    # (Opzionale) stepLengthReduction
    if SDPB_STEP:
        args = ["--stepLengthReduction", SDPB_STEP] + args
    return args

def run_sdpb(run_dir: Path, sdp_path: Path) -> int:
    # ripulisci out
    outdir = run_dir / SDPB_OUT_DIR
    if outdir.exists(): shutil.rmtree(outdir, ignore_errors=True)
    outdir.mkdir(parents=True, exist_ok=True)

    base = sdpb_base_args(sdp_path, run_dir)

    # MPI command inside container
    inner = ["sdpb"] + base
    if SDPB_NP > 1:
        mpicmd = ["mpirun", "--allow-run-as-root", "-np", str(SDPB_NP), "--bind-to", "none"]
        if USE_HWTHREADS: mpicmd += ["--use-hwthread-cpus"]
        inner = mpicmd + ["sdpb"] + base

    docker_cmd = ["docker","run","--rm",
                  "-v", f"{run_dir}:{DOCKER_WORK}",
                  "-e","OMP_NUM_THREADS=1",
                  DOCKER_IMAGE] + inner
    rc, _dt = time_block(run_logged, docker_cmd)
    return rc

def check_once(c: float, delta_trial: float) -> str:
    run_dir = ensure_run_dir(c, delta_trial)
    log.info("[CHECK] c=%.6g Δ=%.6g → run_dir=%s", c, delta_trial, run_dir)

    # cache
    st = parse_status(run_dir)
    if st != "UNKNOWN":
        log.info("CACHE HIT: %s", st)
        return st

    # params per Mathematica
    write_params_for_nb(run_dir, c, delta_trial)

    # wolfram → genera out_raw.json (idealmente nel run_dir)
    rc = run_wolfram(run_dir)
    if rc != 0:
        log.error("wolframscript fallito (rc=%d).", rc)
        return "UNKNOWN"

    # converter + pmp2sdp
    try:
        sdp_path = run_cleaner_and_pmp2sdp(run_dir, PMP2SDP_PREC)
    except Exception as e:
        log.error("Pulizia/Conversione fallita: %s", e)
        return "UNKNOWN"

    # sdpb
    rc = run_sdpb(run_dir, sdp_path)
    status = parse_status(run_dir)
    log.info("SDPB rc=%d, status=%s", rc, status)
    return status

# ──────────────────────────────────────────────────────────────────────────────
# Separazione + bisezione
# ──────────────────────────────────────────────────────────────────────────────
def separate_interval(c: float, lo: float, hi: float):
    slo = check_once(c, lo)
    shi = check_once(c, hi)

    # estendi/riduci se non separato
    tries_up = 0
    while slo == shi and tries_up < 3:
        hi *= 1.5
        log.warning("[c=%.6g] Intervallo non separato (slo=shi=%s). Estendo hi→ %.6f", c, slo, hi)
        shi = check_once(c, hi)
        tries_up += 1

    tries_down = 0
    while slo == shi and tries_down < 3 and lo > 1e-6:
        lo = max(1e-6, lo/2)
        log.warning("[c=%.6g] Ancora non separato. Riduco lo→ %.6f", c, lo)
        slo = check_once(c, lo)
        tries_down += 1

    if slo == shi:
        raise RuntimeError(f"[c={c}] impossibile inizializzare la bisezione (stesso stato a lo/hi: {slo}).")

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
            # caso "instabile": allarghiamo leggermente l'intervallo
            hi *= 1.1
            lo = max(1e-6, lo/1.1)
    return 0.5*(lo+hi)

# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    # Limitare BLAS/OpenMP
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    log.info("=== Avvio bisezione (Nmax=%d) ===", NMAX)
    results = []  # <<— LISTA (niente .append su dict!)
    for c in C_LIST:
        log.info("=== Bisezione su c=%.6g ===", c)
        log.warning("Intervallo iniziale [%.6f, %.6f], tol=%g", DELTA_LO, DELTA_HI, TOL)
        try:
            lo, slo, hi, shi = separate_interval(c, DELTA_LO, DELTA_HI)
            bound = bisect(c, lo, slo, hi, shi)
            log.info("[c=%.6g] Bound ≈ %.6f", c, bound)
            results.append({"c": c, "bound": bound, "Nmax": NMAX})
        except Exception as e:
            log.error("Errore su c=%.6g: %s", c, e)
            # Non scriviamo bound=0: saltiamo il record se non separato
            continue

    RESULTS_JSON.write_text(json.dumps(results, indent=2))
    log.info("Salvato: %s", RESULTS_JSON)
    log.info("Fine.")
