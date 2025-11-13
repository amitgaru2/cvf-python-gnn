SERVER_MACHINES=("manaslu5.uwyo.edu:8098" "manaslu6.uwyo.edu:8098" "manaslu7.uwyo.edu:8098" "manaslu8.uwyo.edu:8098")
# SERVER_MACHINES=("manaslu6.uwyo.edu:8098" "manaslu8.uwyo.edu:8098")
CLIENT_MACHINES=("yangra1.uwyo.edu" "yangra2.uwyo.edu" "yangra3.uwyo.edu" "yangra4.uwyo.edu" "yangra5.uwyo.edu" "yangra6.uwyo.edu" "yangra7.uwyo.edu" "yangra8.uwyo.edu" "yangra9.uwyo.edu" "yangra10.uwyo.edu" "yangra11.uwyo.edu" \
                 "manaslu1.uwyo.edu" "manaslu2.uwyo.edu" "manaslu3.uwyo.edu" "manaslu4.uwyo.edu" \
                 "manaslu9.uwyo.edu" "manaslu10.uwyo.edu" "manaslu11.uwyo.edu" "manaslu12.uwyo.edu")

# CLIENT_MACHINES=("yangra2.uwyo.edu" "yangra3.uwyo.edu" \
#                  )

# CLIENT_MACHINES=("manaslu1.uwyo.edu" "manaslu2.uwyo.edu" "manaslu3.uwyo.edu" "manaslu4.uwyo.edu")
SERVER_MACHINES_ENV=$(IFS=';'; echo "${SERVER_MACHINES[*]}")
NUM_SERVERS=${#SERVER_MACHINES[@]}
NUM_CLIENTS=${#CLIENT_MACHINES[@]}
