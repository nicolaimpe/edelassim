import yaml
from vortex import toolbox

# import cen

toolbox.active_now = True


with open("config_obs_put.yaml", "r") as file:
    # Charger le contenu du fichier en tant que dictionnaire Python
    config = yaml.safe_load(file)
if config["kind"] in ("MeteorologicalForcing"):
    config.update(
        local=config["local"].replace(
            ".nc",
            f"_{config['experiment']}_{config['geometry']}_{config['datebegin']}_{config['dateend']}.nc",
        )
    )
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

print("Footprints", config)
tb = toolbox.output(**config)

# tb.quickview()
# tb.check()
# print(tb.locate())
# tb.get()
