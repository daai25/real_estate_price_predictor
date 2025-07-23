import React, { useState } from "react";
import ImageUploading from 'react-images-uploading';
import LocationPicker from "../../components/location-picker/LocationPicker";

export default function Predictor() {
    const [mode, setMode] = useState("rental");

    const handleModeChange = (newMode) => setMode(newMode);

    const [images, setImages] = useState([]);
    const maxNumber = 10;
    // For stable drag highlight
    const [dragActive, setDragActive] = useState(false);

    const [location, setLocation] = useState({
        address: "",
        zip: "",
        city: "",
        region: "",
        lat: null,
        lon: null
    });

    return (
        <div
            style={{
                minHeight: "100vh",
                background: `
          linear-gradient(
            rgba(0, 0, 0, 0.4),
            rgba(0, 0, 0, 0.7)
          ),
          url('https://avantecture.com/wp-content/uploads/2021/10/Bruderhaus-Nr-2-aussen-13.jpg') center/cover no-repeat
        `,
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
                            onClick={() => handleModeChange(m)}
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
                    onSubmit={e => e.preventDefault()}
                >
                    {/* Location picker section - full width */}
                    <div style={{ width: "100%" }}>
                        <LocationPicker onSelect={(loc) => setLocation(loc)} />
                        <p style={{ fontSize: "0.9rem", marginTop: "0.5rem", color: "lightgray" }}>
                            Selected: {location.address || "Click a location on the map"}
                        </p>
                    </div>

                    {/* Input fields in a row */}
                    <div style={{ display: "flex", gap: "1rem", width: "100%", flexWrap: "wrap" }}>
                        {[
                            { label: "Rooms", name: "rooms", type: "number" },
                            { label: "Area (sqm)", name: "area", type: "number" },
                            { label: "Floor", name: "floor", type: "number" },
                        ].map((f) => (
                            <input
                                key={f.name}
                                placeholder={f.label}
                                type={f.type}
                                name={f.name}
                                style={{
                                    flex: 1,
                                    minWidth: "150px",
                                    padding: "0.6rem",
                                    borderRadius: "6px",
                                    border: "none",
                                    fontSize: "1rem",
                                }}
                            />
                        ))}
                    </div>

                    {/* Features checkboxes section */}
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
                                gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))",
                                gap: "1rem",
                                width: "100%",
                                padding: "1.5rem",
                                backgroundColor: "rgba(255, 255, 255, 0.1)",
                                borderRadius: "8px",
                                border: "1px solid rgba(255, 255, 255, 0.2)"
                            }}
                        >
                            {[
                                "Has Balcony",
                                "Is New",
                                "Has View",
                                "Has Garden",
                                "Has Parking",
                                "Has Air Conditioning",
                            ].map((feature) => (
                                <label
                                    key={feature}
                                    style={{
                                        display: "flex",
                                        alignItems: "center",
                                        fontSize: "0.95rem",
                                        color: "white",
                                        cursor: "pointer",
                                        padding: "0.5rem",
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
                                        style={{ 
                                            marginRight: "0.75rem",
                                            width: "16px",
                                            height: "16px",
                                            cursor: "pointer"
                                        }} 
                                    />
                                    {feature}
                                </label>
                            ))}
                        </div>
                    </div>

                    {/* Image upload full width */}
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

                {/* Submit button */}
                <div style={{ marginTop: "2rem" }}>
                    <button
                        type="submit"
                        style={{
                            padding: "0.8rem 2rem",
                            fontSize: "1rem",
                            fontWeight: "600",
                            color: "#000",
                            backgroundColor: "white",
                            border: "none",
                            borderRadius: "8px",
                            cursor: "pointer",
                            boxShadow: "0 2px 6px rgba(0,0,0,0.3)",
                        }}
                    >
                        Predict Price
                    </button>
                </div>
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
    );
}
