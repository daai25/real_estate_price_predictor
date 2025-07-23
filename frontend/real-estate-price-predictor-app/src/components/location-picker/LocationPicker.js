import React, { useState, useEffect, useRef } from "react";
import { MapContainer, TileLayer, Marker, useMapEvents } from "react-leaflet";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import "./LocationPicker.css";

// Fix default Leaflet marker paths
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png",
  iconRetinaUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png",
  shadowUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png"
});

function LocationMarker({ onLocationSelect, position }) {
  useMapEvents({
    click(e) {
      const { lat, lng } = e.latlng;
      onLocationSelect([lat, lng]);
    }
  });

  return position ? <Marker position={position} /> : null;
}

function CustomSearch({ onLocationSelect, mapRef }) {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [showResults, setShowResults] = useState(false);
  const [selectedAddress, setSelectedAddress] = useState("");

  const searchAddress = async (searchQuery) => {
    if (!searchQuery || searchQuery.length < 3) {
      setResults([]);
      setShowResults(false);
      return;
    }

    setIsLoading(true);
    try {
      const response = await fetch(
        `https://nominatim.openstreetmap.org/search?format=json&addressdetails=1&limit=5&q=${encodeURIComponent(searchQuery)}`
      );
      const data = await response.json();
      setResults(data);
      setShowResults(true);
    } catch (error) {
      console.error("Search error:", error);
      setResults([]);
    }
    setIsLoading(false);
  };

  const handleResultClick = (result) => {
    const lat = parseFloat(result.lat);
    const lng = parseFloat(result.lon);

    if (mapRef.current) {
      mapRef.current.setView([lat, lng], 15);
    }

    onLocationSelect([lat, lng], result);
    setQuery(result.display_name);
    setSelectedAddress(result.display_name);
    setShowResults(false);
    setResults([]);
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      if (results.length > 0) {
        handleResultClick(results[0]);
      }
    }
    if (e.key === "Escape") {
      setShowResults(false);
      setResults([]);
    }
  };

  useEffect(() => {
    if (query && query !== selectedAddress && query.length >= 3) {
      const timeoutId = setTimeout(() => searchAddress(query), 300);
      return () => clearTimeout(timeoutId);
    } else if (query !== selectedAddress) {
      setResults([]);
      setShowResults(false);
    }
  }, [query, selectedAddress]);

  return (
    <div className="search-container">
      <input
        type="text"
        value={query}
        onChange={(e) => {
          setQuery(e.target.value);
          if (e.target.value !== selectedAddress) {
            setSelectedAddress("");
          }
        }}
        onKeyDown={handleKeyDown}
        placeholder="🔍 Search for an address..."
      />

      {isLoading && <div className="search-loading">Searching...</div>}

      {showResults && results.length > 0 && (
        <div className="result-list">
          {results.map((result, index) => (
            <div
              key={index}
              className="result-item"
              onClick={() => handleResultClick(result)}
            >
              {result.display_name}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default function LocationPicker({ onSelect }) {
  const [position, setPosition] = useState(null);
  const mapRef = useRef(null);

  const handleLocationSelect = (coords, result = null) => {
    setPosition(coords);

    if (result) {
      const address = result.address || {};
      onSelect({
        address: result.display_name,
        zip: address.postcode || "",
        city: address.city || address.town || address.village || "",
        region: address.state || address.county || "",
        lat: coords[0],
        lon: coords[1]
      });
    } else {
      fetch(
        `https://nominatim.openstreetmap.org/reverse?format=jsonv2&lat=${coords[0]}&lon=${coords[1]}`
      )
        .then((res) => res.json())
        .then((data) => {
          const address = data.address || {};
          onSelect({
            address: data.display_name,
            zip: address.postcode || "",
            city: address.city || address.town || address.village || "",
            region: address.state || address.county || "",
            lat: coords[0],
            lon: coords[1]
          });
        });
    }
  };

  return (
    <div className="location-picker-container">
      <MapContainer
        center={[47.3769, 8.5417]}
        zoom={13}
        style={{ height: "100%", width: "100%" }}
        ref={mapRef}
      >
        <TileLayer
          attribution='&copy; <a href="http://osm.org/copyright">OpenStreetMap</a> contributors'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />
        <LocationMarker
          onLocationSelect={handleLocationSelect}
          position={position}
        />
      </MapContainer>
      <CustomSearch onLocationSelect={handleLocationSelect} mapRef={mapRef} />
    </div>
  );
}
