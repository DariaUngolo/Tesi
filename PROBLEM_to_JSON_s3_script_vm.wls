(* ::Package:: *)

(* ================= PROBLEM_to_JSON_s3_script.wls (VM FAST + SANITIZE + CONSERVATIVE FALLBACK) ================= *)
(* PATCHES:
   - Crea/Cancella FALLBACK.flag così il Python ignora i fallback (UNKNOWN)
   - Mantiene struttura e numerica del tuo script originale
*)

(* ---- Libreria SDPB ---- *)
Get["/home/azureuser/bootstrap_data/SDPB.m"];

(* ---- Parallelismo e MKL ---- *)
Quiet[LaunchKernels[]];
(*processi separati del Kernel Wolfram) che eseguono in parallelo le chiamate tipo ParallelTable, ParallelMap*)
SetSystemOptions["MKLThreads" -> Max[1, Min[$ProcessorCount, 8]]];(*quanti kernel usare per ogni thread*)

(* ---- Impostazioni numeriche ---- *)
$HistoryLength = 0; (*non vengono salvate le ultime espressioni out(n)*)
$MaxExtraPrecision = 200;
$Assumptions = Element[x, Reals]; (*assume che x sia reale*)

timestamp[] := DateString[Now, {"YYYY", "-", "MM", "-", "DD", " ", "HH", ":", "mm", ":", "ss"}];
log[msg_]  := Print["[WLS] ", timestamp[], " | ", msg];
warn[msg_] := Print["[WLS][WARN] ",  timestamp[], " | ", msg];
err[msg_]  := Print["[WLS][ERROR] ", timestamp[], " | ", msg];

log["start"];

(* ----- FALLBACK FLAG ----- *)
workdir    = Directory[]; (*cartella dove eseguo lo script*)
fallbackFlagPath = FileNameJoin[{workdir, "FALLBACK.flag"}];
clearFallbackFlag[] := If[FileExistsQ[fallbackFlagPath], DeleteFile[fallbackFlagPath]];
markFallback[] := Put["FALLBACK_USED", fallbackFlagPath];
clearFallbackFlag[];

(* ---- Lettura parametri ---- *)
(*{ "c": 0.8, "DeltaTrial": 1.1, "N_max": 4 }*)
paramsPath = FileNameJoin[{workdir, "params_for_nb_S3.json"}];
If[!FileExistsQ[paramsPath], err["manca params_for_nb_S3.json in " <> workdir]; Exit[1]];
params   = Quiet @ Import[paramsPath, "RawJSON"];


(*spacchetta i parametri nelle varie variabili*)
cdef     = N[params["c"]];
myDelta  = SetPrecision[params["DeltaTrial"], 80];
Nmax     = params["N_max"];
If[!NumericQ[cdef] || !NumericQ[myDelta] || !IntegerQ[Nmax] || Nmax < 0,
  err["Parametri non validi: " <> ToString[params, InputForm]]; Exit[1];
];

myPrecision = 80; (*numero di cifre decimali*)
degreeMax  = 2 Nmax; (*costruiamo polinomi fino a grando 2Nmax*)

(* ---- Switch derivate ---- *)
USE_SERIESCOEFF = True; (*Serve a scegliere tra due metodi equivalenti per ottenere le derivate all’origine*)

(* ---- Definizioni base ---- *)
Chih[z_, c_, h_]  := Exp[z/4]  Exp[-2 Pi Exp[z]  (h + (1 - c)/24)];
Chihb[zb_, c_, h_]:= Exp[zb/4] Exp[-2 Pi Exp[zb] (h + (1 - c)/24)];
ChihFix[z_, h_]   := Chih[z,  cdef, h];
ChihbFix[zb_, h_] := Chihb[zb, cdef, h];

Chi0[z_,  c_]  := Exp[z/4]  Exp[-2 Pi Exp[z]  (1 - c)/24] (1 - Exp[-2 Pi Exp[z]]);
Chi0b[zb_, c_] := Exp[zb/4] Exp[-2 Pi Exp[zb] (1 - c)/24] (1 - Exp[-2 Pi Exp[zb]]);
Chi0Fix[z_]    := Chi0[z,  cdef];
Chi0bFix[zb_]  := Chi0b[zb, cdef];

(* Blocchi *)
F [z_, zb_, x_, s_] := ChihFix[z,  (2 s + myDelta + x)/2]*ChihbFix[zb, (myDelta + x)/2];
(*trasformazione modulare al blocco F*)
SF[z_, zb_, x_, s_] := ChihFix[-z, (2 s + myDelta + x)/2]*ChihbFix[-zb,(myDelta + x)/2];

(* Fattore esponenziale per scalare *)
expFactor[x_, s_] := Exp[-2 Pi (-(1/96) + x/2) - 2 Pi (-(1/96) + 1/2 (2 s + x))];

(* Vacuum row uguale al caso di prima*)
F00[z_, zb_]  := Chi0Fix[z]*Chi0bFix[zb];
SF00[z_, zb_] := Chi0Fix[-z]*Chi0bFix[-zb];

(* ---- Derivate ---- *)
(*due metodi diversi per fare le derivate uno con serie di taylore uno usando D*)
dCoeff[m_, n_, expr_] := SeriesCoefficient[expr, {z, 0, m}, {zb, 0, n}] * m! * n!;
If[TrueQ[USE_SERIESCOEFF],
  deriv[z_,  zb_, x_, s_] := Flatten @ Table[dCoeff[m, n,  F[z,  zb, x, s]], {m, 0, Nmax}, {n, 0, Nmax}];
  derivSF[z_, zb_, x_, s_] := Flatten @ Table[dCoeff[m, n, SF[z, zb, x, s]], {m, 0, Nmax}, {n, 0, Nmax}],
  deriv[z_,  zb_, x_, s_] := Flatten @ Table[D[F [z, zb, x, s],  {z, m}, {zb, n}] /. {z -> 0, zb -> 0}, {m, 0, Nmax}, {n, 0, Nmax}];
  derivSF[z_, zb_, x_, s_] := Flatten @ Table[D[SF[z, zb, x, s], {z, m}, {zb, n}] /. {z -> 0, zb -> 0}, {m, 0, Nmax}, {n, 0, Nmax}];
];

L  = (Nmax + 1)^2;(*lunghezza del vettore derivato*)

(* Vacuum derivates memoized *)
(*Calcola una sola volta*)
dersF00  := dersF00  = Flatten @ Table[D[F00[z, zb],  {z, m}, {zb, n}] /. {z -> 0, zb -> 0}, {m, 0, Nmax}, {n, 0, Nmax}];
dersSF00 := dersSF00 = Flatten @ Table[D[SF00[z, zb], {z, m}, {zb, n}] /. {z -> 0, zb -> 0}, {m, 0, Nmax}, {n, 0, Nmax}];

(* Safe numeric + memoization per s *)
(*converte il numero a my precision e converte indeterminato/infinito in 0*)
safeN[val_] := Module[{v = Quiet @ Check[N[val, myPrecision], Indeterminate]},
  Which[
    v === Indeterminate || v === ComplexInfinity || MatchQ[v, DirectedInfinity[_]], 0,
    True, v
  ]
];

(*memorizza i vettori riscalati*)
DF[x_, s_]  := DF[x, s]  = N[ Map[safeN,  deriv [z, zb, x, s]/expFactor[x, s]], myPrecision];
DSF[x_, s_] := DSF[x, s] = N[ Map[safeN, derivSF[z, zb, x, s]/expFactor[x, s]], myPrecision];

(*vettore di 0*)
ZeroVec := ConstantArray[0, L];

(* ---- ANOMALIE ---- *)
omega = Exp[2 Pi I/3];   kZ2 = 0;
(*kZ2 = 0,1*)
(*omega=1,Exp[2 Pi I/3],Exp[4 Pi I/3]*)


(* ---- Regole di spin ---- *)
AllowedSpinsZ3[om_] := Which[
  om == 1,             {1/2, 1/3, 2/3, 1},
  om == Exp[2 Pi I/3], {1/2, 1/9, 4/9, 7/9},
  om == Exp[4 Pi I/3], {1/2, 2/9, 5/9, 8/9}
];
AllowedSpinsZ2[k_]  := Module[{base = {0, 1/2, 1, 3/2}}, Rationalize[Mod[k/4 + base, 1]]];

(*regole di spin normali senza anomalie*)
(*da capire metto solo gli spin interi?*)
AllowedSpinsId[]    := { 1, 2,3,4};
(*AllowedSpinsId[]    := {1/2, 1, 3/2, 2};*)

(* ---- Vettori di settore ---- *)
(*azzera il rumore numerico*)

(*da CAPIRE MEGLIO perchè più settori hanno questo termine*)
V10[x_] := {
  Chop[dersSF00 - 3/6 dersF00, 10^-40],
  Chop[dersSF00-3/6 dersF00,          10^-40],
  Chop[dersSF00-2 dersF00,          10^-40],
  Chop[ZeroVec,              10^-40],
  Chop[-3/2 dersF00,              10^-40]
};

V1[x_, s_]  := Module[{d = DF[x, s], ds = DSF[x, s]},
  {Chop[ds - 1/6 d, 10^-40], Chop[-1/6 d, 10^-40], Chop[-2/3 d, 10^-40], ZeroVec, Chop[-d, 10^-40]}
];
V1P[x_, s_] := Module[{d = DF[x, s], ds = DSF[x, s]},
  {Chop[-1/6 d, 10^-40], Chop[ds - 1/6 d, 10^-40], Chop[-2/3 d, 10^-40], ZeroVec, Chop[-d, 10^-40]}
];
V2[x_, s_]  := Module[{d = DF[x, s], ds = DSF[x, s]},
  {Chop[-1/6 d, 10^-40], Chop[-1/6 d, 10^-40], Chop[ds - 2/3 d, 10^-40], ZeroVec, Chop[ 1/2 d, 10^-40]}
];
VA[x_, s_]  := Module[{d = DF[x, s], ds = DSF[x, s]},
  {Chop[-1/2 d, 10^-40], Chop[ 1/2 d, 10^-40], ZeroVec, Chop[ds, 10^-40], ZeroVec}
];
VS[x_, s_]  := Module[{d = DF[x, s], ds = DSF[x, s]},
  {Chop[-1/3 d, 10^-40], Chop[-1/3 d, 10^-40], Chop[ 2/3 d, 10^-40], ZeroVec, Chop[ds, 10^-40]}
];

(* ---- Liste per settore ---- *)
(*mettiamo in parallelo tutte le righe del vettore*)
listV1 [x_] := ParallelTable[V1 [x, s], {s, AllowedSpinsId[]}];
listV1P[x_] := ParallelTable[V1P[x, s], {s, AllowedSpinsId[]}];
listV2 [x_] := ParallelTable[V2 [x, s], {s, AllowedSpinsId[]}];
listVA[x_]  := ParallelTable[VA [x, s], {s, AllowedSpinsZ2[kZ2]}];
listVS[x_]  := ParallelTable[VS [x, s], {s, AllowedSpinsZ3[omega]}];


(*ad ogni lista associa un settore*)
bySectorLists[x_] := <|"1"->listV1[x], "1p"->listV1P[x], "2"->listV2[x], "A"->listVA[x], "S"->listVS[x]|>;

(* ---- Conversione polinomi → coefficienti + sanificazione ---- *)
(*converte in numeri con precisione myprecision e azzera rumori*)
(*fino a degree max*)
PolyToArrayClean[expr_] := Module[{poly, coeffs},
  poly   = Quiet @ Check[Expand[Together[expr]], 0];
  coeffs = Table[Coefficient[poly, x, k], {k, 0, degreeMax}];
  N[Chop[coeffs, 10^-40], myPrecision]
];

(*trasforma numero vicino a 0 in 0*)
cleanZero[b_] := If[Chop[b, 10^-40] == 0, 0, b];


(*converte numeri in stringhe*)
toNumString[val_] := Module[{b = N[val, myPrecision], r, s},
  r = Round[b];
  s = If[Chop[b - r, 10^-40] == 0,
    ToString[r],
    StringReplace[ToString @ NumberForm[b, {Infinity, 50}, ExponentFunction -> (Null &)], " " -> ""]
  ];
  StringReplace[s, "-0" -> "0"]
];

prefattore = DampedRational[1, {}, SetPrecision[1/E, myPrecision], x];


(*se tutti i coefficientisono piccolissimi li vede come blocco vuoto*)
NearlyZeroCoeffsQ[coeffs_List, thr_: 10^-40] := (Max[Abs[coeffs]] < thr);


(*prende le 5 componenti le concatene e per ogni compoente associa la lista dei coeffciienti x*)
(*li concatena e li converte in stringhe e mette dentro matrice 1x1*)
SectorSpin1x1ConcatAllSafe[vec5_] := Module[{conc, coeffsNum, coeffsStr},
  conc      = Join @@ vec5;
  coeffsNum = PolyToArrayClean /@ conc;
  If[AllTrue[coeffsNum, NearlyZeroCoeffsQ], Return[$Failed]]; (* blocco nullo → scarta *)
  coeffsStr = (toNumString / @ (cleanZero /@ #)) & /@ coeffsNum;
  {{coeffsStr}}
];

(* ======================== Fallback conservativo ========================= *)
fallbackSDP[] := Module[{L0, Lsp, objectiveVec, normFunctional, polsDummy, pref},
  L0 = (Nmax + 1)^2; Lsp = 5 L0;
  objectiveVec = ConstantArray[0, Lsp];
  normFunctional = ConstantArray[0, Lsp];

  pref = DampedRational[1, {}, 1, x];  (* prefattore innocuo *)
  polsDummy = {
    PositiveMatrixWithPrefactor[
      <|"prefactor"->pref, "polynomials"->{{{{"0"}}}}|>  (* 1x1 polinomio identicamente 0 *)
    ]
  };

  SDP[objectiveVec, normFunctional, polsDummy]
];

safeWritePMP[out_, sdp_] := Module[{},
  Check[
    WritePmpJson[out, sdp, myPrecision],
    (* se pure l'export fallisce, ritenta con il fallback minimo *)
    warn["PROBLEMA_CALCOLO: WritePmpJson fallita, esporto fallback."];
    markFallback[];
    WritePmpJson[out, fallbackSDP[], myPrecision]
  ]
];

(* =================== Costruzione con cattura errori ==================== *)
problemHappened = False;
outFile = FileNameJoin[{workdir, "out_raw_S3.json"}];

result = Check[
  Module[{assoc, sectorOrder, polsRaw, polsClean, pols, L0, Lsp, objectiveVec, normFunctionalVec, normFunctional},
    sectorOrder = {"1","1p","2","A","S"};
    assoc = bySectorLists[x];

    polsRaw = Flatten @ Table[
      ParallelMap[SectorSpin1x1ConcatAllSafe, assoc[sec]],
      {sec, sectorOrder}
    ];

    polsClean = DeleteCases[polsRaw, $Failed];
    If[Length[polsClean] == 0,
      warn["PROBLEMA_CALCOLO: tutti i blocchi nulli dopo sanificazione (Nmax="<>ToString[Nmax]<>"). Uso fallback."];
      markFallback[];
      safeWritePMP[outFile, fallbackSDP[]];
      Throw["done"]
    ];

    pols = PositiveMatrixWithPrefactor[<|"prefactor"->prefattore, "polynomials"->#|>] & /@ polsClean;

    L0  = L;  Lsp = 5 L0;
    objectiveVec = ConstantArray[0, Lsp];

    (* NOTA: normFunctional = riga vacuum (identità) — verifica che il segno sia quello atteso *)
    normFunctionalVec = Join @@ V10[x];
    normFunctional    = N[Chop[normFunctionalVec, 10^-40], myPrecision];
    If[Length[normFunctional] =!= Lsp,
      warn["PROBLEMA_CALCOLO: normFunctional lunghezza " <> ToString[Length[normFunctional]] <>
           " attesa " <> ToString[Lsp] <> " → fallback."];
      markFallback[];
      safeWritePMP[outFile, fallbackSDP[]];
      Throw["done"]
    ];

    log["Esporto JSON → " <> outFile <> " …"];
    safeWritePMP[outFile, SDP[objectiveVec, normFunctional, pols]]; (*ECCO DOVE CREO IL FILE JSONQquindi*)
  ],
  (* Catch generale di errori in costruzione *)
  problemHappened = True;
  warn["PROBLEMA_CALCOLO intercettato (costruzione). Esporto fallback."];
  markFallback[];
  safeWritePMP[outFile, fallbackSDP[]];
];

If[result === "done", problemHappened = True];

If[TrueQ[problemHappened],
  warn["PROBLEMA_CALCOLO: generato JSON CONSERVATIVO (trattare come UNKNOWN nella bisezione)."],
  Print["[OK] out_raw_S3.json generato con c=", cdef, "  DeltaTrial=", myDelta, "  Nmax=", Nmax,
        "  | Kernels: ", $KernelCount];
];

Exit[0];


(* COME SONO FATTI I NOSTRI VINCOLI 
"positive_matrices": [
  { ... },  // settore 1, spin 1/2
  { ... },  // settore 1, spin 1
  { ... },  // settore 1, spin 3/2
  { ... },  // settore 1, spin 2
  { ... },  // settore 1p, spin 1/2
  ...
  { ... },  // settore S, spin 1
]

*)

