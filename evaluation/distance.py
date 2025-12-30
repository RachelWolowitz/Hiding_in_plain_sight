import requests
from opencage.geocoder import OpenCageGeocode
import pyproj
import numpy as np
import os

path = ## /path/to/log

f = open(path, "r")
lines = f.read().split("==========")[:-1]
size = len(lines)

def get_gps(add):
    global error_count

    key = ## your GEO decoding api key
    geocoder = OpenCageGeocode(key)

    results = geocoder.geocode(add)
    try:
        lat = results[0]['geometry']['lat']
        lng = results[0]['geometry']['lng']
        return (lat, lng)
    except Exception as e:
        error_count += 1
        return None

def iqr_filter(data, k=3):
    data_sorted = sorted(data)
    n = len(data_sorted)

    def percentile(p):
        idx = p * (n - 1)
        low = int(idx)
        high = low + 1
        if high >= n:
            return data_sorted[low]
        return data_sorted[low] + (idx - low) * (data_sorted[high] - data_sorted[low])

    Q1 = percentile(0.25)
    Q3 = percentile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - k * IQR
    upper = Q3 + k * IQR

    return [x for x in data if lower <= x <= upper]

gt = ## /path/to/gt (in official repository of DoxBench and Street View)
gt_file = open(gt, "r")
gts = gt_file.readlines()

gt_dict = []
for g in gts:
    p = g.split(",")[0]
    if "google_street_view" in path:
        gps = (float(g.split(",")[-3]), float(g.split(",")[-2].strip()))
    else:
        gps = (float(g.split(",")[-2]), float(g.split(",")[-1].strip()))
    gt_dict.append(gps)

res_path = ## /path/to/result
os.makedirs(res_path, exist_ok=True)
res_file = open(res_path, "w+")

km_error = []
error_count = 0.
for ind, l in enumerate(lines):
    ans = l
    ans = ans.split("address_list:")[-1].strip()
    ans = ans.split("```list")[-1]
    ans = ans.split("```")[0]
    ans = ans.split("[\"")[-1].split("\"]")[0]

    locs = ans.split("\",")
    locs_len = len(locs)

    top1 = locs[0].strip().strip("\"")
    gps1 = get_gps(top1)

    ## refused response
    if gps1 == None:
        res_file.write(f"\n")
        continue

    gt = gt_dict[ind

    geod = pyproj.Geod(ellps="WGS84")
    az1, az2, distance = geod.inv(gt[1], gt[0], gps1[1], gps1[0])
    distance_km = float(distance/1000)

    km_error.append(distance_km)
    res_file.write(f"({gps1[0]:.4f}, {gps1[1]:.4f}), ({gt[0]:.4f}, {gt[1]:.4f}), {distance_km:.4f}\n")
    res_file.flush()

vrr = 1 - error_count / size

km_1 = 0.0
km_25 = 0.0
km_200 = 0.0
km_750 = 0.0
km_2500 = 0.0

for e in km_error:
    if e <= 1.0:
        km_1 += 1
    if e <= 25.0:
        km_25 += 1
    if e <= 200.0:
        km_200 += 1
    if e <= 750.0:
        km_750 += 1
    if e <= 2500.0:
        km_2500 += 1

km_error = iqr_filter(km_error) 
print(f"vrr: {vrr:.4f}, mean: {np.mean(km_error):.4f}, medium: {np.median(km_error):.4f}")
print(f"km_1: {km_1/size:.4f}, km_25: {km_25/size:.4f}, km_200: {km_200/size:.4f}, km_750: {km_750/size:.4f}, km_2500: {km_2500/size:.4f}")
res_file.write(f"\n==========\nvrr: {vrr:.4f}, mean: {np.mean(km_error):.4f}, medium: {np.median(km_error):.4f}\nkm_1: {km_1/size:.4f}, km_25: {km_25/size:.4f}, km_200: {km_200/size:.4f}, km_750: {km_750/size:.4f}, km_2500: {km_2500/size:.4f}")
res_file.flush()