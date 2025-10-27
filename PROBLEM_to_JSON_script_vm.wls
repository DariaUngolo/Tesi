(* === PROBLEM_to_JSON_script_vm.wls (headless, VM) === *)
(* Legge params_for_nb.json e scrive out_raw.json nella stessa cartella. *)
workdir = DirectoryName[$InputFileName];
SetDirectory[workdir];

(* Carica SDPB.m dalla posizione fissa richiesta *)
Get["/home/azureuser/bootstrap_data/SDPB.m"];

(* Precisione numerica *)
myPrecision = 50;

(* Parametri da JSON: c arriva DAL PYTHON, qui non lo impostiamo. *)
paramsPath = FileNameJoin[{workdir, "params_for_nb.json"}];
If[FileExistsQ[paramsPath],
  params  = Import[paramsPath, "RawJSON"];
  cdef    = params["c"];  (* passato dal Python *)
  myDelta = SetPrecision[params["DeltaTrial"], myPrecision];
  smax    = params["s_max"];
  Nmax    = params["N_max"];
,
  (* fallback minimale *)
  cdef    = Rationalize[5/4];
  myDelta = SetPrecision[0, myPrecision];
  smax    = 2; Nmax = 2;
];

(* --- Codice del problema (invariato) --- *)
Chih[z_, c_, h_] := Exp[z/4] Exp[-2 Pi Exp[z] (h + (1 - c)/24)];
Chihb[zb_, c_, hb_] := Exp[zb/4] Exp[-2 Pi Exp[zb] (hb + (1 - c)/24)];
ChihFix[z_, h_] := Chih[z, cdef, h];
ChihbFix[zb_, hb_] := Chihb[zb, cdef, hb];

Chi0[z_, c_] := Exp[z/4]*Exp[-2 Pi Exp[z]*(1 - c)/24]*(1 - Exp[-2 Pi Exp[z]]);
Chi0b[zb_, c_] := Exp[zb/4]*Exp[-2 Pi Exp[zb]*(1 - c)/24]*(1 - Exp[-2 Pi Exp[zb]]);
Chi0Fix[z_] := Chi0[z, cdef];
Chi0bFix[zb_] := Chi0b[zb, cdef];

F[z_, zb_, x_, s_] :=
  ChihFix[z, (2 s + myDelta + x)/2]*ChihbFix[zb, (myDelta + x)/2] +
  ChihFix[zb, (myDelta + x)/2]*ChihbFix[z, (2 s + myDelta + x)/2];

derivOdd =
  Flatten@Table[
    If[OddQ[m + n],
      D[F[z, zb, x, s], {z, m}, {zb, n}] /. {z -> 0, zb -> 0},
      Nothing
    ],
    {m, 0, Nmax}, {n, 0, Nmax}
  ];

L = Length[derivOdd];

expFactor = Exp[-2 Pi (-(1/96) + x/2) - 2 Pi (-(1/96) + 1/2 (2 s + x))];
prefattore = DampedRational[1, {}, 1/E, x];

F00[z_, zb_] := Chi0Fix[z]*Chi0bFix[zb];
dersOddF00 =
  Flatten@Table[
    If[OddQ[m + n],
      D[F00[z, zb], {z, m}, {zb, n}] /. {z -> 0, zb -> 0},
      Nothing
    ],
    {m, 0, Nmax}, {n, 0, Nmax}
  ];

norm = dersOddF00;
normNum = N[norm, myPrecision];

derivOddFactorized = derivOdd/expFactor;
derivOddFactorizedSimpl = Simplify /@ derivOddFactorized;
AllderivativeOdd = N[Expand[#], myPrecision] & /@ derivOddFactorizedSimpl;

allDerivOdd = Table[AllderivativeOdd /. s -> sVal, {sVal, 0, smax}];

degreesAll = Exponent[#, x] & /@ Flatten[allDerivOdd];
degreesAll = degreesAll /. -Infinity -> 0;
maxDegree = Max[Prepend[degreesAll, 0]];

padCoeffs[poly_] := PadRight[CoefficientList[poly, x], maxDegree + 1, 0];
cleanZero[b_] := If[Chop[b, 10^-40] == 0, 0, b];

pols =
  Table[
    Module[{polysS, coeffNumS},
      polysS    = allDerivOdd[[sVal]];
      coeffNumS = (cleanZero /@ padCoeffs[#]) & /@ polysS;  (* numeri, non stringhe *)
      PositiveMatrixWithPrefactor[
        <|
          "prefactor"   -> prefattore,
          "polynomials" -> {{coeffNumS}}
        |>
      ]
    ],
    {sVal, 1, smax}
  ];

objNum = ConstantArray[0, L];

filePath = FileNameJoin[{workdir, "out_raw.json"}];
WritePmpJson[filePath, SDP[objNum, normNum, pols], myPrecision];
Print["OK out_raw.json: ", FileExistsQ[filePath]];