import glob
from pathlib import Path

import pandas as pd
import yaml
from vortex import toolbox

# import cen

toolbox.active_now = True

config_folder = "../config/vortex_configs"
with open(f"{config_folder}/config_prep_background.yaml", "r") as file:
    # Charger le contenu du fichier en tant que dictionnaire Python
    config = yaml.safe_load(file)
# if config["kind"] in ("MeteorologicalForcing"):
#     config.update(
#         local=config["local"].replace(
#             ".nc",
#             f"_{config['experiment']}_{config['geometry']}_{config['datebegin']}_{config['dateend']}.nc",
#         )
#     )
# elif config["kind"] in ("PREP"):
#     config.update(
#         local=config["local"].replace(
#             ".nc",
#             f"_{config['experiment']}_{config['geometry']}_{config['date']}.nc",
#         )
#     )


# if "member" in config:
#     config.update(
#         local=config["local"].replace(
#             ".nc",
#             f"_member_{config['member'][0]}_{config['member'][-1]}.nc",
#         )
#     )

# for filename in glob.glob(
#     "/home/imperatoren/work/edelweiss_assimilation/observations/grandesrousses250m/meteofrance/soda/*.nc"
# ):
#     # for date in pd.date_range(start="2021/10/01", end="2022/07/31", freq="D"):
#     # date = date + pd.DateOffset(hours=12)
#     mf, fsc, l3, platform, date_str = Path(config["filename"]).name.split("_")
#     # date = pd.Timestamp.strptime("%Y%m%d%H")
#     config["date"] = date_str.split(".")[0]

#     # new_filename = f"{Path(config['filename']).parent}/{mf}_{fsc}_{l3}_{platform}_{date_str}.nc"
#     config["filename"] = filename
#     print("Footprints", config)
#     tb = toolbox.output(**config)


tb = toolbox.input(**config)

# tb.quickview()
# tb.check()
# print(tb.locate())
# tb.get()
