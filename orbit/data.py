selected_sv = ['PG01', 'PE03', 'PR06', 'PC06', 'PC25', 'PJ02']
####### PG01 #######
# G080 2024:352:00000 0000:000:00000 G01
# G080 2024-242A 62339 GPS-IIIA Launched 2024-12-17; NAVSTAR 83

##### PE03 ########
#E212 2016:322:00000 0000:000:00000 E03 
# E212 2016-069B  41860 GAL-2           Launched 2016-11-17; GALILEO 16 (26C)

####### PR06 ########
#R804 2025:358:19350 0000:000:00000 R06 start time from tracking data
# R804 2025-042A  63130 GLO-K2          Launched 2025-03-02; COSMOS 2584

####### PC06 #########
#C220 2026:100:57600 0000:000:00000 C06 [PR04], NABU 20260006
# C220 2019-023A  44204 BDS-3I          Launched 2019-04-20; BEIDOU 3 IGSO-1

####### PC25 ##########
#C212 2018:236:00000 0000:000:00000 C25 
#C212 2018-067B  43603 BDS-3M-SECM-A   Launched 2018-08-24; BEIDOU 3M12

######### PJ02 #########
#J002 2017:152:00000 0000:000:00000 J02 [PR01]
#J002 2017-028A  42738 QZS-2I          Launched 2017-06-01; MICHIBIKI-2


#https://cddis.nasa.gov/archive/gnss/products/2419/
#https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=tle
#https://files.igs.org/pub/station/general/igs_satellite_metadata.snx
#https://www.space-track.org/#/queryBuilder
import georinex as gr

file_path = 'orbit_raw/raw_sp3_data/COD0MGXFIN_20261370000_01D_05M_ORB.SP3'

ds = gr.load(file_path)

print(ds)

print("\nLista satelitów w tym pliku:")
print(ds.sv.values)