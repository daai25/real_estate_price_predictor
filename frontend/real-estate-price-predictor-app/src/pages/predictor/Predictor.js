import React, { useState } from "react";
import ImageUploading from 'react-images-uploading';
import LocationPicker from "../../components/location-picker/LocationPicker";

export default function Predictor() {
    const [mode, setMode] = useState("rental");
    const [images, setImages] = useState([]);
    const maxNumber = 10;
    const [dragActive, setDragActive] = useState(false);
    const [prediction, setPrediction] = useState(null); // NEW: prediction display
    const [isLoading, setIsLoading] = useState(false); // NEW: loading state

    const [location, setLocation] = useState({
        address: "",
        zip: "",
        city: "",
        region: "",
        lat: null,
        lon: null
    });

    const [form, setForm] = useState({ // NEW: form state
        rooms: "",
        area: "",
        floor: "",
        has_balcony: false,
        is_new: false,
        has_view: false,
        has_garden: false,
        has_parking: false,
        has_air_conditioning: false,
    });

    const handleSubmit = async (e) => { // NEW: API call
        e.preventDefault();
        setIsLoading(true); // Start loading
        setPrediction(null); // Clear previous prediction

        const formData = new FormData();
        formData.append("mode", mode);
        formData.append("rooms", form.rooms);
        formData.append("area", form.area);
        formData.append("floor", form.floor);
        formData.append("zip_code", location.zip);
        formData.append("city", location.city);
        formData.append("region", location.region);
        formData.append("lat", location.lat);
        formData.append("lon", location.lon);
        formData.append("availability_date", new Date().toISOString().split("T")[0]);
        formData.append("description_cluster", "0");

        formData.append("has_balcony", form.has_balcony ? 1 : 0);
        formData.append("is_new", form.is_new ? 1 : 0);
        formData.append("has_view", form.has_view ? 1 : 0);
        formData.append("has_garden", form.has_garden ? 1 : 0);
        formData.append("has_parking", form.has_parking ? 1 : 0);
        formData.append("has_air_conditioning", form.has_air_conditioning ? 1 : 0);

        // Send multiple images to match the updated API
        if (images.length > 0) {
            images.forEach((image, index) => {
                if (image.file) {
                    formData.append("images", image.file);
                }
            });
        }

        console.log("Sending form data:", {
            mode,
            rooms: form.rooms,
            area: form.area,
            floor: form.floor,
            location,
            images: images.length,
            features: {
                has_balcony: form.has_balcony,
                is_new: form.is_new,
                has_view: form.has_view,
                has_garden: form.has_garden,
                has_parking: form.has_parking,
                has_air_conditioning: form.has_air_conditioning,
            }
        });

        try {
            console.log("Sending request to API...");
            const response = await fetch("http://localhost:5000/predict", {
                method: "POST",
                body: formData,
            });
            
            console.log("Response received:", response.status, response.statusText);
            
            const result = await response.json();
            console.log("API Response data:", result);
            
            if (response.ok) {
                console.log("Success! Prediction:", result.prediction);
                setPrediction(result.prediction);
                // Removed alert - prediction is displayed below the form
            } else {
                console.error("API Error:", result);
                alert(`Prediction failed: ${result.error || 'Unknown error'}`);
            }
        } catch (err) {
            console.error("Network/API error:", err);
            alert("An error occurred while contacting the prediction server. Make sure the Flask API is running on localhost:5000.");
        } finally {
            setIsLoading(false); // Stop loading regardless of success/failure
        }
    };

    return (
        <>
            <style>
                {`
                    @keyframes spin {
                        0% { transform: rotate(0deg); }
                        100% { transform: rotate(360deg); }
                    }
                `}
            </style>
            <div
                style={{
                    minHeight: "100vh",
                    background: `linear-gradient(rgba(0, 0, 0, 0.4), rgba(0, 0, 0, 0.7)), url('https://avantecture.com/wp-content/uploads/2021/10/Bruderhaus-Nr-2-aussen-13.jpg') center/cover no-repeat`,
                    fontFamily: '"Helvetica Neue", Helvetica, Arial, sans-serif',
                    color: "white",
                    display: "flex",
                    flexDirection: "column",
                    alignItems: "center",
                    textAlign: "center",
                    padding: "2rem",
                }}
            >
            <main
                style={{
                    width: "100%",
                    maxWidth: "900px",
                    borderRadius: "12px",
                    padding: "2rem"
                }}
            >
                <h1 style={{ fontSize: "2.5rem", marginBottom: "1.5rem" }}>
                    Property Price Predictor
                </h1>

                {/* Mode toggle */}
                <div
                    style={{
                        display: "flex",
                        justifyContent: "center",
                        gap: "1rem",
                        marginBottom: "2rem",
                    }}
                >
                    {["rental", "purchase"].map((m) => (
                        <button
                            key={m}
                            onClick={() => setMode(m)}
                            style={{
                                padding: "0.6rem 1.5rem",
                                fontSize: "0.95rem",
                                fontWeight: "600",
                                borderRadius: "999px",
                                border: "1px solid white",
                                backgroundColor: mode === m ? "white" : "transparent",
                                color: mode === m ? "#000" : "white",
                                cursor: "pointer",
                                transition: "0.3s",
                            }}
                        >
                            {m === "rental" ? "Rental" : "Purchase"}
                        </button>
                    ))}
                </div>

                {/* Form layout */}
                <form
                    style={{
                        display: "flex",
                        flexDirection: "column",
                        gap: "2rem",
                        width: "100%"
                    }}
                    onSubmit={handleSubmit} // MODIFIED
                >
                    <div style={{ width: "100%" }}>
                        <LocationPicker onSelect={(loc) => setLocation(loc)} />
                        <p style={{ fontSize: "0.9rem", marginTop: "0.5rem", color: "lightgray" }}>
                            Selected: {location.address || "Click a location on the map"}
                        </p>
                    </div>

                    <div style={{ display: "flex", gap: "1rem", width: "100%", flexWrap: "wrap" }}>
                        {[
                            { label: "Rooms", name: "rooms", type: "number" },
                            { label: "Area (sqm)", name: "area", type: "number" },
                            { label: "Floor", name: "floor", type: "number" },
                        ].map((f) => (
                            <div key={f.name} style={{ flex: 1, minWidth: "150px" }}>
                                <label style={{
                                    display: "block",
                                    marginBottom: "0.5rem",
                                    fontSize: "0.9rem",
                                    color: "white",
                                    fontWeight: "500"
                                }}>
                                    {f.label}
                                </label>
                                <input
                                    type={f.type}
                                    name={f.name}
                                    value={form[f.name] || ""}
                                    onChange={(e) => setForm({ ...form, [f.name]: e.target.value })}
                                    style={{
                                        width: "100%",
                                        padding: "0.6rem",
                                        borderRadius: "6px",
                                        border: "none",
                                        fontSize: "1rem",
                                        boxSizing: "border-box",
                                        textAlign: "center"
                                    }}
                                />
                            </div>
                        ))}
                    </div>

                    <div style={{ width: "100%" }}>
                        <h3 style={{
                            fontSize: "1.1rem",
                            marginBottom: "1rem",
                            color: "white",
                            fontWeight: "500",
                            textAlign: "left"
                        }}>
                            Property Features
                        </h3>
                        <div
                            style={{
                                display: "grid",
                                gridTemplateColumns: "1fr 1fr",
                                gap: "1rem",
                                width: "100%",
                                padding: "0"
                            }}
                        >
                            {[
                                "Has Balcony",
                                "Is New",
                                "Has View",
                                "Has Garden",
                                "Has Parking",
                                "Has Air Conditioning",
                            ].map((featureLabel) => {
                                const featureKey = featureLabel
                                    .toLowerCase()
                                    .replace(/ /g, "_");

                                return (
                                    <label
                                        key={featureLabel}
                                        style={{
                                            display: "flex",
                                            alignItems: "center",
                                            fontSize: "1rem",
                                            color: "white",
                                            cursor: "pointer",
                                            padding: "0.75rem",
                                            borderRadius: "4px",
                                            transition: "background-color 0.2s",
                                            backgroundColor: "transparent"
                                        }}
                                        onMouseEnter={(e) => {
                                            e.target.style.backgroundColor = "rgba(255, 255, 255, 0.1)";
                                        }}
                                        onMouseLeave={(e) => {
                                            e.target.style.backgroundColor = "transparent";
                                        }}
                                    >
                                        <input
                                            type="checkbox"
                                            checked={form[featureKey] || false}
                                            onChange={(e) =>
                                                setForm({ ...form, [featureKey]: e.target.checked })
                                            }
                                            style={{
                                                marginRight: "1rem",
                                                width: "20px",
                                                height: "20px",
                                                cursor: "pointer"
                                            }}
                                        />
                                        {featureLabel}
                                    </label>
                                );
                            })}
                        </div>
                    </div>

                    <div style={{ width: "100%" }}>
                        <ImageUploading
                            multiple
                            value={images}
                            onChange={(imageList) => setImages(imageList)}
                            maxNumber={maxNumber}
                            dataURLKey="data_url"
                        >
                            {({
                                imageList,
                                onImageUpload,
                                onImageRemoveAll,
                                onImageUpdate,
                                onImageRemove,
                                dragProps,
                                isDragging
                            }) => (
                                <div
                                    {...dragProps}
                                    onDragEnter={e => { setDragActive(true); if (dragProps.onDragEnter) dragProps.onDragEnter(e); }}
                                    onDragLeave={e => { setDragActive(false); if (dragProps.onDragLeave) dragProps.onDragLeave(e); }}
                                    onDrop={e => { setDragActive(false); if (dragProps.onDrop) dragProps.onDrop(e); }}
                                    style={{
                                        border: dragActive ? "2.5px solid #2196f3" : "2px dashed #ccc",
                                        boxShadow: dragActive ? "0 0 0 3px #2196f366" : undefined,
                                        background: dragActive ? "rgba(33,150,243,0.08)" : "rgba(255,255,255,0.05)",
                                        transition: "border 0.2s, background 0.2s, box-shadow 0.2s",
                                        padding: "1rem",
                                        borderRadius: "6px",
                                    }}
                                >
                                    <button
                                        onClick={e => { e.preventDefault(); onImageUpload(); }}
                                        style={{
                                            padding: "0.6rem 1.5rem",
                                            fontWeight: "600",
                                            backgroundColor: "white",
                                            color: "black",
                                            border: "none",
                                            borderRadius: "6px",
                                            cursor: "pointer",
                                            marginBottom: "1rem",
                                        }}
                                    >
                                        Click or Drag & Drop Images
                                    </button>
                                    {imageList.length > 0 && (
                                        <button
                                            onClick={e => { e.preventDefault(); onImageRemoveAll(); }}
                                            style={{
                                                marginLeft: "1rem",
                                                padding: "0.6rem 1rem",
                                                fontSize: "0.9rem",
                                                background: "transparent",
                                                border: "1px solid white",
                                                color: "white",
                                                borderRadius: "6px",
                                                cursor: "pointer"
                                            }}
                                        >
                                            Remove All
                                        </button>
                                    )}
                                    <div
                                        style={{
                                            display: "flex",
                                            gap: "1rem",
                                            flexWrap: "wrap",
                                            marginTop: "1rem",
                                            justifyContent: "center",
                                        }}
                                    >
                                        {imageList.map((image, index) => (
                                            <div key={index} style={{ position: "relative" }}>
                                                <img
                                                    src={image.data_url}
                                                    alt=""
                                                    width="100"
                                                    height="100"
                                                    style={{ borderRadius: "6px", objectFit: "cover" }}
                                                />
                                                <div
                                                    style={{
                                                        display: "flex",
                                                        justifyContent: "center",
                                                        gap: "0.3rem",
                                                        marginTop: "0.3rem",
                                                    }}
                                                >
                                                    <button
                                                        onClick={e => { e.preventDefault(); onImageUpdate(index); }}
                                                        style={{
                                                            fontSize: "0.75rem",
                                                            padding: "0.2rem 0.5rem",
                                                            backgroundColor: "white",
                                                            color: "black",
                                                            border: "none",
                                                            borderRadius: "4px",
                                                            cursor: "pointer",
                                                        }}
                                                    >
                                                        Update
                                                    </button>
                                                    <button
                                                        onClick={e => { e.preventDefault(); onImageRemove(index); }}
                                                        style={{
                                                            fontSize: "0.75rem",
                                                            padding: "0.2rem 0.5rem",
                                                            backgroundColor: "#ff4d4f",
                                                            color: "white",
                                                            border: "none",
                                                            borderRadius: "4px",
                                                            cursor: "pointer",
                                                        }}
                                                    >
                                                        Remove
                                                    </button>
                                                </div>
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            )}
                        </ImageUploading>
                    </div>
                </form>

                <div style={{ 
                    marginTop: "2rem", 
                    display: "flex", 
                    justifyContent: "center", 
                    width: "100%" 
                }}>
                    <button
                        type="submit"
                        onClick={handleSubmit}
                        disabled={isLoading}
                        style={{
                            padding: "0.8rem 2rem",
                            fontSize: "1rem",
                            fontWeight: "600",
                            color: isLoading ? "#666" : "#000",
                            backgroundColor: isLoading ? "#f0f0f0" : "white",
                            border: "none",
                            borderRadius: "8px",
                            cursor: isLoading ? "not-allowed" : "pointer",
                            boxShadow: "0 2px 6px rgba(0,0,0,0.3)",
                            display: "flex",
                            alignItems: "center",
                            justifyContent: "center",
                            gap: "0.5rem",
                            transition: "all 0.2s",
                        }}
                    >
                        {isLoading && (
                            <div
                                style={{
                                    width: "16px",
                                    height: "16px",
                                    border: "2px solid #666",
                                    borderTop: "2px solid transparent",
                                    borderRadius: "50%",
                                    animation: "spin 1s linear infinite",
                                }}
                            />
                        )}
                        {isLoading ? "Predicting..." : "Predict Price"}
                    </button>
                </div>

                {prediction && (
                    <p style={{ 
                        marginTop: "1.5rem", 
                        fontSize: "1.5rem", 
                        color: "#e0fcffff",
                        fontWeight: "bold",
                        textShadow: "0 2px 4px rgba(0,0,0,0.3)"
                    }}>
                        💰 Predicted Price: CHF {prediction.toLocaleString()}
                    </p>
                )}
            </main>

            <footer
                style={{
                    textAlign: "center",
                    padding: "1rem",
                    fontSize: "0.85rem",
                    opacity: 0.7,
                }}
            >
                Michael · Josh · Enmanuel · Alessandro
            </footer>
        </div>
        </>
    );
}
