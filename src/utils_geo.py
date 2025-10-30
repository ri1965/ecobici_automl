import numpy as np

# Haversine (metros)
def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    p1 = np.radians(lat1); p2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1); dl = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(p1)*np.cos(p2)*np.sin(dl/2)**2
    return 2*R*np.arcsin(np.sqrt(a))

def nearest_station(df_stations, click_lat, click_lon):
    d = haversine_m(click_lat, click_lon, df_stations["lat"].values, df_stations["lon"].values)
    idx = int(np.argmin(d))
    return df_stations.iloc[idx], float(d[idx])