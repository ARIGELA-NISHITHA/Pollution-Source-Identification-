import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import folium

# Load data
df = pd.read_csv('pollution_data.csv')
print(df.head())

# Calculate pollution score (basic example)
df['pollution_score'] = df[['pm25', 'pm10', 'no2', 'so2']].mean(axis=1)

# Identify high pollution areas (threshold = 100)
high_pollution = df[df['pollution_score'] > 100]

# Plot pollution levels
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='longitude', y='latitude', size='pollution_score', hue='pollution_score', palette='coolwarm', legend=False)
plt.title("Pollution Levels by Location")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.grid(True)
plt.show()

# Optional: Cluster locations to identify pollution source zones
coords = df[['latitude', 'longitude']]
kmeans = KMeans(n_clusters=3, random_state=0).fit(coords)
df['cluster'] = kmeans.labels_

# Visualize on map using Folium
map_center = [df['latitude'].mean(), df['longitude'].mean()]
pollution_map = folium.Map(location=map_center, zoom_start=12)

for _, row in df.iterrows():
    folium.CircleMarker(
        location=[row['latitude'], row['longitude']],
        radius=5,
        popup=f"Score: {row['pollution_score']:.1f}",
        color='red' if row['pollution_score'] > 100 else 'green',
        fill=True,
    ).add_to(pollution_map)

pollution_map.save("pollution_map.html")
print("Pollution map saved as pollution_map.html")

# Optional: Print likely source clusters
print("\nPotential Pollution Hotspots (Cluster Centers):")
print(kmeans.cluster_centers_)
