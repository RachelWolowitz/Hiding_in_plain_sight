import requests
import pyproj
import numpy as np

ori_res_path = ## /path/to/clean_image_response
ori = open(ori_res_path, "r")
ori_lines = ori.readlines()

adv_res_path = ## /path/to/protected_image_response
adv = open(adv_res_path, "r")
adv_lines = adv.readlines()

name = adv_res_path.split("/")[-1].split(".txt")[0]
assert len(ori_lines) == len(adv_lines)

km_error = []
for o, a in zip(ori_lines, adv_lines):
    if o == "\n" or a == "\n":
        continue
    o_gps = o.split("),")[0]
    a_gps = a.split("),")[0]

    o_lat = float(o_gps.split(",")[0].split("(")[-1])
    o_lng = float(o_gps.split(",")[1].split(")")[0])
    a_lat = float(a_gps.split(",")[0].split("(")[-1])
    a_lng = float(a_gps.split(",")[1].split(")")[0])

    geod = pyproj.Geod(ellps="WGS84")
    az1, az2, distance = geod.inv(o_lng, o_lat, a_lng, a_lat)
    distance_km = float(distance/1000)

    km_error.append(distance_km)

mean_km_error = np.mean(km_error)
print(f"mean: {mean_km_error:.4f}")