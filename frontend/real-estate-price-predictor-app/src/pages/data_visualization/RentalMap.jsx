import React, { useEffect, useState } from "react";
import { MapContainer, TileLayer, Marker, Popup } from "react-leaflet";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import "./RentalMap.css";

import rentaldata from "./rentaldata.json";

// Fix Leaflet marker icon paths
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png",
  iconRetinaUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png",
  shadowUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png"
});

export default function RentalMap() {
  const [geoResults, setGeoResults] = useState([]);

  useEffect(() => {
    const filtered = rentaldata.filter(
      (item) => item.latitude && item.longitude
    );
    setGeoResults(filtered);
  }, []);

  return (
    <MapContainer center={[47.3769, 8.5417]} zoom={12} className="rental-map">
      <TileLayer
        attribution='&copy; <a href="https://osm.org/">OpenStreetMap</a> contributors'
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
      />
      {geoResults.map((rental, index) => (
        <Marker
          key={index}
          position={[rental.latitude, rental.longitude]}
        >
          <Popup>
            <strong>{rental.title}</strong><br />
            {rental.address}, {rental.city}<br />
            CHF {rental.price?.toFixed(0)} / month<br />
            Rooms: {rental.rooms}<br />
            {rental.has_balcony && "🌿 Balcony"}<br />
            {rental.image_urls?.[0] && (
              <img
                src={rental.image_urls[0]}
                alt="preview"
                width="150"
                style={{ marginTop: "8px", borderRadius: "4px" }}
              />
            )}
          </Popup>
        </Marker>
      ))}
    </MapContainer>
  );
}