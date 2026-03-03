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

with open("config_prep_analysis.yaml", "w") as file:
    # Charger le contenu du fichier en tant que dictionnaire Python
    yaml.dump(config, file)
