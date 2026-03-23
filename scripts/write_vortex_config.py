import footprints
import yaml

# config = dict(
#     geometry="grandesrousses",
#     kind="MeteorologicalForcing",
#     model="safran",
#     datebegin="2005080106",
#     dateend="2006080106",
#     date="[dateend]",
#     experiment="reanalysis2020.2@lafaysse",
#     filename="FORCING.nc",
#     namespace="vortex.multi.fr",
#     block="meteo",
#     vapp="s2m",
#     vconf="[geometry:tag]",
#     namebuild="flat@cen",
# )
dateend_map = dict(datebegin={"2021080206": "2022022612", "2022022612": "2022080106"})
config = dict(
    kind="PREP",  # "prep"
    vapp="edelweiss",
    vconf="[geometry:tag]",
    geometry="GrandesRousses250m",
    # datebegin=["2021080206", "2022022612"],  # supprimer pour prep
    # dateend=dateend_map,  # supprimer pour prep
    date="2022022612",  # une date pour le prep (date assim)
    member=footprints.util.rangex(0, 16, 1),
    model="surfex",
    experiment="TEST_assim",
    username="imperatoren",
    block="prep/analysis",  # prep/analysis ou #prep/background
    local="/home/imperatoren/work/edelweiss_assimilation/simulations/edelweiss/[geometry:tag]/[experiment]/mb[member]/prep/analysis/PREP_[date:ymdh].nc",  # PREP_[date].nc
    namebuild="flat@cen",
    namespace="vortex.multi.fr",
)

config = dict(
    filename="/home/imperatoren/work/edelweiss_assimilation/data/snow_cover/viirs_composite/soda/MF_FSC_VJ1_L3_20220226.nc",
    date="2022022612",  # '2018012312', '2018031612', '2022022612', '2022050112'
    vapp="viirs",
    vconf="[geometry:tag]",
    experiment="assim_viirs",
    geometry="GrandesRousses250m",
    kind="SnowObservations",
    model="surfex",
    block="",
    scope="viirs",
    namespace="vortex.multi.fr",
    namebuild="flat@cen",
    fatal=True,
)

with open("config_obs_put.yaml", "w") as file:
    # Charger le contenu du fichier en tant que dictionnaire Python
    yaml.dump(config, file)
