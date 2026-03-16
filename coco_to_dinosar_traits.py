#!/usr/bin/env python3
"""
coco_to_dinosar_traits.py  (v2 — dual-format)

Convert COCO JSON exports from BOTH:
  A) Descriptron GBIF Annotator (web-based, HTML) — both exportCOCO() and generateCOCOJSON()
  B) Descriptron desktop (Python/Tkinter)

into the TSV formats expected by DINOSARv2 v24 multi-modal training:

  1) trait.tsv   → for --trait-tsv  (one-hot categorical attributes + mask areas)
  2) morph.tsv   → for --morph-tsv  (continuous measurements from masks/lines/scale)

Handles two COCO schemas from the GBIF annotator:

  exportCOCO() — full Descriptron schema:
    - annotations[].attributes        (region-level, from ...regionMeta spread)
    - annotations[].instance_attributes   (per-instance overrides)
    - annotations[].is_trait_only     (attribute-only records, no geometry)
    - annotations[].is_line + line_points (measurement polylines)
    - annotations[].is_scale_bar      (scale bar annotations)
    - images[].gbif_occurrence_id     (GBIF link → specimen_id)
    - attribute_vocabulary            (full vocab from template → complete one-hot)
    - info.scale_bar / scale          (calibration)

  generateCOCOJSON() — simpler schema:
    - annotations[].region_attributes    (region-level)
    - annotations[].instance_attributes  (per-instance)
    - info.scale_bar / scale             (calibration)

Attribute encoding:
  Multi-class nominal (texture, color, sculpture, shape, setae, etc.)
    → one-hot binary: cat_{region}_{attr}_{value}
  If attribute_vocabulary is present, ALL possible values get columns
  (even if not annotated), ensuring consistent column sets across specimens.

Usage:
  python coco_to_dinosar_traits.py \\
    --coco-dir /path/to/coco_jsons/ \\
    --out-trait-tsv trait.tsv \\
    --out-morph-tsv morph.tsv

  python coco_to_dinosar_traits.py \\
    --coco-file /path/to/combined.json \\
    --out-trait-tsv trait.tsv
"""

import argparse, csv, json, math, os, sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from collections import defaultdict


def parse_specimen_id(file_name, mode="auto", gbif_occurrence_id="", coco_info=None):
    if mode in ("auto","occurrence") and gbif_occurrence_id:
        return gbif_occurrence_id.strip()
    if mode in ("auto","occurrence") and coco_info:
        occ_id = coco_info.get("occurrence","")
        if isinstance(occ_id,str) and occ_id.strip(): return occ_id.strip()
    stem = Path(file_name).stem
    if mode == "full_stem": return stem
    if "_" in stem:
        parts = stem.rsplit("_",1)
        sfx = parts[1].upper()
        views = {"H","D","P","L","V","A","HEAD","DORSAL","PROFILE","LATERAL","VENTRAL","ANTERIOR","POSTERIOR"}
        if sfx in views or (len(sfx)<=2 and sfx.isalpha()): return parts[0]
    return stem


def _get_annotation_attributes(ann):
    attrs = {}
    if "attributes" in ann and isinstance(ann["attributes"],dict):
        for k,v in ann["attributes"].items():
            if v and isinstance(v,str): attrs[k]=v
    if "region_attributes" in ann and isinstance(ann["region_attributes"],dict):
        for k,v in ann["region_attributes"].items():
            if v and isinstance(v,str): attrs[k]=v
    if "instance_attributes" in ann and isinstance(ann["instance_attributes"],dict):
        for k,v in ann["instance_attributes"].items():
            if k!="_notes" and v and isinstance(v,str): attrs[k]=v
    return attrs


def _line_length_px(pts):
    t=0.0
    for i in range(1,len(pts)):
        p0,p1=pts[i-1],pts[i]
        if isinstance(p0,(list,tuple)) and isinstance(p1,(list,tuple)):
            t+=math.sqrt((p1[0]-p0[0])**2+(p1[1]-p0[1])**2)
    return t


def extract_from_coco(coco, specimen_id_override=None, specimen_id_mode="auto"):
    cat_by_id={}
    for c in coco.get("categories",[]): cat_by_id[c["id"]]={"name":c.get("name",""),"label":c.get("label",c.get("name",""))}
    img_by_id={img["id"]:img for img in coco.get("images",[])}
    coco_info=coco.get("info",{})
    scale_px=None; scale_unit=None
    si=coco_info.get("scale_bar",coco.get("scale",None))
    if si and isinstance(si,dict): scale_px=si.get("px_per_unit"); scale_unit=si.get("unit")
    attr_vocab=coco.get("attribute_vocabulary",{})
    anns_by_img=defaultdict(list)
    for a in coco.get("annotations",[]): anns_by_img[a.get("image_id",1)].append(a)
    records=[]
    for img_id,img_info in img_by_id.items():
        fn=img_info.get("file_name",f"image_{img_id}")
        gbif_occ=img_info.get("gbif_occurrence_id","")
        sid=specimen_id_override or parse_specimen_id(fn,specimen_id_mode,gbif_occ,coco_info)
        regions={}; lines=[]
        local_scale=scale_px
        for ann in anns_by_img.get(img_id,[]):
            cid=ann.get("category_id",-1); ci=cat_by_id.get(cid,{"name":"","label":""})
            rn=ci["label"] or ci["name"] or f"region_{cid}"
            attrs=_get_annotation_attributes(ann)
            if ann.get("is_scale_bar"):
                s=ann.get("scale_px_per_unit")
                if s: local_scale=s; scale_unit=ann.get("scale_unit",scale_unit)
            if ann.get("is_line") and ann.get("line_points"):
                lines.append({"region":rn,"length_px":_line_length_px(ann["line_points"])})
            area=ann.get("area",0.0); bbox=ann.get("bbox",[0,0,0,0])
            bw=bbox[2] if len(bbox)>2 else 0; bh=bbox[3] if len(bbox)>3 else 0
            is_to=ann.get("is_trait_only",False)
            if rn not in regions: regions[rn]={"attributes":{},"area_px":0.0,"bbox_w_px":0.0,"bbox_h_px":0.0,"instances":[],"has_geometry":False}
            for k,v in attrs.items():
                if v and k not in regions[rn]["attributes"]: regions[rn]["attributes"][k]=v
            if not is_to and area>regions[rn]["area_px"]:
                regions[rn]["area_px"]=area; regions[rn]["bbox_w_px"]=bw; regions[rn]["bbox_h_px"]=bh; regions[rn]["has_geometry"]=True
            regions[rn]["instances"].append(attrs)
        records.append({"specimen_id":sid,"image_file":fn,"scale_px_per_unit":local_scale,"scale_unit":scale_unit,"regions":regions,"line_measurements":lines})
    return records, attr_vocab


def records_to_trait_tsv(all_records, attr_vocab, out_path, include_area=True):
    observed=defaultdict(set); region_names=set()
    for rec in all_records:
        for rn,rd in rec["regions"].items():
            region_names.add(rn)
            for an,av in rd["attributes"].items():
                if av: observed[f"{rn}:{an}"].add(str(av))
    if attr_vocab:
        for rn in region_names:
            for an,vals in attr_vocab.items():
                k=f"{rn}:{an}"
                if k not in observed: observed[k]=set()
                for v in vals: observed[k].add(str(v))
    cat_cols=[]
    for k in sorted(observed):
        rn,an=k.split(":",1)
        for v in sorted(observed[k]):
            col=f"cat_{rn}_{an}_{v}".replace(" ","_").replace("-","_").replace("/","_")
            cat_cols.append((col,rn,an,v))
    area_cols=[f"area_{rn}_px" for rn in sorted(region_names)] if include_area else []
    fnames=["specimen_id"]+[c[0] for c in cat_cols]+area_cols
    by_spec=defaultdict(list)
    for r in all_records: by_spec[r["specimen_id"]].append(r)
    rows=[]
    for sid in sorted(by_spec):
        recs=by_spec[sid]; row={"specimen_id":sid}
        for cn,rn,an,av in cat_cols:
            found=False
            for rec in recs:
                rd=rec["regions"].get(rn,{})
                if isinstance(rd,dict):
                    if str(rd.get("attributes",{}).get(an,""))==av: found=True; break
                    for inst in rd.get("instances",[]):
                        if str(inst.get(an,""))==av: found=True; break
                if found: break
            row[cn]=1 if found else 0
        if include_area:
            for rn in sorted(region_names):
                mx=0.0
                for rec in recs:
                    rd=rec["regions"].get(rn,{})
                    if isinstance(rd,dict): mx=max(mx,rd.get("area_px",0.0))
                row[f"area_{rn}_px"]=round(mx,1) if mx>0 else ""
        rows.append(row)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".",exist_ok=True)
    with open(out_path,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fnames,delimiter="\t"); w.writeheader()
        for row in rows: w.writerow({k:row.get(k,"") for k in fnames})
    print(f"✓ Wrote {out_path}")
    print(f"  {len(rows)} specimens × {len(cat_cols)} categorical + {len(area_cols)} area columns")
    if attr_vocab: print(f"  attribute_vocabulary expanded: {list(attr_vocab.keys())}")


def records_to_morph_tsv(all_records, out_path):
    region_names=set(); line_regions=set()
    for rec in all_records:
        region_names.update(rec["regions"].keys())
        for lm in rec.get("line_measurements",[]): line_regions.add(lm["region"])
    region_names=sorted(region_names); line_regions=sorted(line_regions)
    mcols=[]
    for rn in region_names: mcols.extend([f"{rn}_area_real",f"{rn}_bbox_w_real",f"{rn}_bbox_h_real",f"{rn}_aspect_ratio"])
    for rn in line_regions: mcols.append(f"{rn}_line_length_real")
    fnames=["specimen_id"]+mcols
    by_spec=defaultdict(list)
    for r in all_records: by_spec[r["specimen_id"]].append(r)
    rows=[]; n_cal=0
    for sid in sorted(by_spec):
        recs=by_spec[sid]; row={"specimen_id":sid}
        scale=None
        for rec in recs:
            if rec.get("scale_px_per_unit") and rec["scale_px_per_unit"]>0: scale=rec["scale_px_per_unit"]; break
        if scale: n_cal+=1
        for rn in region_names:
            best={"a":0.0,"bw":0.0,"bh":0.0}
            for rec in recs:
                rd=rec["regions"].get(rn,{})
                if isinstance(rd,dict) and rd.get("area_px",0)>best["a"]:
                    best={"a":rd.get("area_px",0.0),"bw":rd.get("bbox_w_px",0.0),"bh":rd.get("bbox_h_px",0.0)}
            if scale and best["a"]>0:
                row[f"{rn}_area_real"]=round(best["a"]/scale**2,6)
                row[f"{rn}_bbox_w_real"]=round(best["bw"]/scale,4)
                row[f"{rn}_bbox_h_real"]=round(best["bh"]/scale,4)
                ar=best["bw"]/best["bh"] if best["bh"]>0 else ""
                row[f"{rn}_aspect_ratio"]=round(ar,4) if isinstance(ar,float) else ""
            else:
                for s in ("_area_real","_bbox_w_real","_bbox_h_real","_aspect_ratio"): row[f"{rn}{s}"]=""
        for rn in line_regions:
            mx=0.0
            for rec in recs:
                for lm in rec.get("line_measurements",[]):
                    if lm["region"]==rn: mx=max(mx,lm["length_px"])
            row[f"{rn}_line_length_real"]=round(mx/scale,4) if scale and mx>0 else ""
        rows.append(row)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".",exist_ok=True)
    with open(out_path,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fnames,delimiter="\t"); w.writeheader()
        for row in rows: w.writerow({k:row.get(k,"") for k in fnames})
    print(f"✓ Wrote {out_path}")
    print(f"  {len(rows)} specimens × {len(mcols)} columns, {n_cal}/{len(rows)} calibrated")
    if line_regions: print(f"  Line measurements: {', '.join(line_regions)}")


def main():
    ap=argparse.ArgumentParser(description="Convert Descriptron COCO JSON → DINOSARv2 v24 trait/morph TSVs")
    src=ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--coco-dir",type=str)
    src.add_argument("--coco-file",type=str)
    ap.add_argument("--out-trait-tsv",default="trait.tsv")
    ap.add_argument("--out-morph-tsv",default=None)
    ap.add_argument("--specimen-id-from",default="auto",choices=["auto","filename","full_stem","occurrence"])
    ap.add_argument("--no-area",action="store_true")
    args=ap.parse_args()
    all_records=[]; all_vocab={}
    if args.coco_dir:
        jfs=sorted(Path(args.coco_dir).glob("*.json"))
        print(f"Loading {len(jfs)} COCO JSONs from {args.coco_dir}...")
        for jf in jfs:
            try:
                c=load_single_coco(str(jf)); recs,voc=extract_from_coco(c,specimen_id_mode=args.specimen_id_from)
                all_records.extend(recs)
                for k,v in voc.items(): all_vocab[k]=sorted(set(all_vocab.get(k,[]))|set(v))
            except Exception as e: print(f"  ⚠ Skipping {jf.name}: {e}")
    else:
        print(f"Loading {args.coco_file}...")
        c=load_single_coco(args.coco_file); all_records,all_vocab=extract_from_coco(c,specimen_id_mode=args.specimen_id_from)
    u=set(r["specimen_id"] for r in all_records)
    print(f"  {len(all_records)} images → {len(u)} specimens")
    if not all_records: print("⚠ No records."); return
    records_to_trait_tsv(all_records,all_vocab,args.out_trait_tsv,not args.no_area)
    if args.out_morph_tsv: records_to_morph_tsv(all_records,args.out_morph_tsv)
    print(f"\nDINOSARv2 v24 usage:  --trait-tsv {args.out_trait_tsv}" + (f"  --morph-tsv {args.out_morph_tsv}" if args.out_morph_tsv else ""))

if __name__=="__main__": main()
