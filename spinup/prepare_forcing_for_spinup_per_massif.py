import os
from typing import Any, Dict
import numpy as np
import xarray as xr
import xml.etree.ElementTree as ET

massif_name = "Grandes-Rousses"
years = list(range(2017, 2018))
save_path = "/home/imperatoren/work/edelweiss_assimilation/"
metadata_file_path = "/home/imperatoren/work/edelweiss_assimilation/snowtools/snowtools/DATA/METADATA.xml"


def parse_latlon_metadata_xml(metedata_xml_filepath: str):
    tree = ET.parse(metedata_xml_filepath)
    root = tree.getroot()
    root.attrib

    massif_dict = {}

    for idx_list_massif in range(len(root[0])):
        massif_dict.update(
            {
                root[0][idx_list_massif][0].text: {
                    root[0][idx_list_massif][2].tag: root[0][idx_list_massif][2].text,  # number
                    root[0][idx_list_massif][3].tag: root[0][idx_list_massif][3].text,  # latCenter
                    root[0][idx_list_massif][4].tag: root[0][idx_list_massif][4].text,  # lonCenter
                },
            }
        )
    return massif_dict


def my_sel(data: xr.Dataset, **kwargs) -> xr.Dataset:
    if len(kwargs) != 1:
        raise NotImplementedError
    else:
        for k, w in kwargs.items():
            data_out = data.sel(Number_of_points=data.data_vars[k] == w)
    return data_out


massif_metadata = parse_latlon_metadata_xml(metadata_file_path)

# out_tmp_files = []
out_datasets = []
for begin_year in years:
    print("Treat year", begin_year, " to ", begin_year + 1)
    # Find the right longitude and latitude from /home/delacroixb/GitHub/snowtools/snowtools/DATA/METADATA.xml
    lat_true = float(massif_metadata[massif_name]["latCenter"])
    lon_true = float(massif_metadata[massif_name]["lonCenter"])

    # Select the massif
    data = xr.open_dataset(
        "/rd/cenfic3/cenmod/era40/vortex/s2m/alp_allslopes/reanalysis/meteo/FORCING_"
        + str(begin_year)
        + "080106_"
        + str(begin_year + 1)
        + "080106.nc"
    )

    data_massif = my_sel(data, massif_number=int(massif_metadata[massif_name]["number"]))

    # Check latitue et longitude
    lat = data_massif.data_vars["LAT"]
    lon = data_massif.data_vars["LON"]

    if np.all(lat == lat[0]) and np.all(lon == lon[0]):
        lat_value = lat[0].values
        lon_value = lon[0].values
    else:
        raise ValueError

    assert round(lat_value.item(), 3) == round(lat_true, 3), (
        f"Latitude attendue {round(lat_true, 3)} et celle recuperee {round(lat_value.item(), 3)}"
    )
    assert round(lon_value.item(), 3) == round(lon_true, 3), (
        f"Latitude attendue {round(lon_true, 3)} et celle recuperee {round(lon_value.item(), 3)}"
    )

    # # Supprimer la dimension massif du dataset selectionne.
    only_data_point = data_massif.drop_dims("massif")
    # print(only_data_point)
    # # Save
    out_tmp_file = save_path + "FORCING_" + str(begin_year) + "080106_" + str(begin_year + 1) + "080106.nc"
    # out_tmp_files.append(out_tmp_file)
    only_data_point.to_netcdf(out_tmp_file)
    out_datasets.append(only_data_point)


# out_datasets = [xr.open_dataset(f) for f in out_tmp_files]
# forcing_tot = xr.concat(out_datasets, dim="time", data_vars="minimal")
# forcing_tot = xr.open_mfdataset(out_tmp_files, data_vars="all")
# forcing_tot.to_netcdf(save_path + "FORCING_" + str(years[0]) + "080106_" + str(years[-1] + 1) + "080106.nc")
# os.remove([f for f in out_tmp_files])
