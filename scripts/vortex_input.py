import footprints
from vortex import toolbox
import cen

# toolbox.active_now = True

# dateend_map = dict(datebegin={"2017080106": "2018080106", "2021080206": "2022080106"})

# pro = toolbox.input(
#     kind="MeteorologicalForcing",
#     vapp="s2m",
#     vconf="[geometry:tag]",
#     geometry="grandesrousses",
#     datebegin="2017080106",
#     dateend="2018080106",
#     date="[dateend]",
#     member=footprints.util.rangex(0, 34, 1),
#     model="safran",
#     experiment="perturb.reanalysis2020.2@lafaysse",
#     local="forcing_s2m_granderousses/mb[member]/FORCING_[datebegin:ymdh]_[dateend:ymdh].nc",
#     namebuild="flat@cen",
#     namespace="vortex.multi.fr",
#     block="meteo",
# )

toolbox.input(
    filename="Pleiades_[date:ymdh].nc",
    date="2019051312",  # '2020050412'
    vapp="Pleiades",
    vconf="[geometry:tag]",
    experiment="CesarDB_AngeH@vernaym",
    geometry="Huez250m",
    kind="SnowObservations",
    model="surfex",
    block="",
    namespace="vortex.multi.fr",
    namebuild="flat@cen",
    fatal=True,
)
