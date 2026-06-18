"""Spot-check the refined education-overlap rule against known cases."""
import sys, io
import pandas as pd
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
f = pd.read_parquet(Path(r"A:\side_hustle\IndiaRuns Hackathon\redrob-ranking\artifacts\features.parquet")).set_index("candidate_id")

cases = {
    "CAND_0026716": "B.Sc@Stanford(2012-15) + PhD@Symbiosis(2012-16)  gap2  EXPECT FLAG",
    "CAND_0061257": "M.E.@IITRoorkee + M.S.@VJTI               gap0  EXPECT RELEASE",
    "CAND_0078002": "M.E.@NITTrichy + B.E.@IITGuwahati          gap1  EXPECT RELEASE",
    "CAND_0052682": "M.E.@NITWarangal + PhD@IITGuwahati         gap1  EXPECT RELEASE",
    "CAND_0066999": "M.S.@IIITBangalore + B.Tech@IITRoorkee     gap1  EXPECT RELEASE",
    "CAND_0084283": "M.Sc@CMU + M.E.@KIIT                       gap0  EXPECT RELEASE",
}
for cid, desc in cases.items():
    if cid in f.index:
        flag = bool(f.loc[cid, "education_anomaly_flag"])
        detail = str(f.loc[cid, "education_anomaly_detail"])
        mark = "FLAG  " if flag else "clear "
        print(f"  {mark} {cid}  {desc}")
        if detail and detail != "nan":
            print(f"            detail: {detail}")
    else:
        print(f"  ???    {cid} not found")
