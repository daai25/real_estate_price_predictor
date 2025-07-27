import React from 'react';

export default function Explorer() {


    // Dummy data for demonstration
    // Fetch data from backend API
    const [data, setData] = React.useState([]);
    const [filters, setFilters] = React.useState({
        city: '',
        region: '',
        minPrice: '',
        maxPrice: '',
        minRooms: '',
        maxRooms: '',
        minArea: '',
        maxArea: '',
        isRental: 'all',
        hasBalcony: 'all',
        hasParking: 'all',
        hasGarden: 'all',
        hasView: 'all',
        hasAirConditioning: 'all'
    });

    React.useEffect(() => {
        console.log("Fetching data from API...");
        fetch("http://localhost:5001/api/properties")
            .then(res => {
                console.log("Response status:", res.status);
                return res.json();
            })
            .then(data => {
                console.log("Received data:", data);
                console.log("Data type:", typeof data);
                console.log("Is array:", Array.isArray(data));
                console.log("Data length:", data?.length);
                
                // Ensure data is an array
                if (Array.isArray(data)) {
                    setData(data);
                    console.log("Set data to state:", data.length, "items");
                } else {
                    console.warn("API response is not an array:", data);
                    setData([]);
                }
            })
            .catch(error => {
                console.error("Error fetching data:", error);
                setData([]);
            });
    }, []);
    console.log(data);
    
    // Use all data instead of pagination
    const dataArray = Array.isArray(data) ? data : [];
    
    // Filter data based on filters
    const filteredData = dataArray.filter(row => {
        // City filter
        if (filters.city && !row.city?.toLowerCase().includes(filters.city.toLowerCase())) {
            return false;
        }
        
        // Region filter
        if (filters.region && !row.region?.toLowerCase().includes(filters.region.toLowerCase())) {
            return false;
        }
        
        // Price filters
        if (filters.minPrice && row.price && row.price < parseInt(filters.minPrice)) {
            return false;
        }
        if (filters.maxPrice && row.price && row.price > parseInt(filters.maxPrice)) {
            return false;
        }
        
        // Rooms filters
        if (filters.minRooms && row.rooms && row.rooms < parseFloat(filters.minRooms)) {
            return false;
        }
        if (filters.maxRooms && row.rooms && row.rooms > parseFloat(filters.maxRooms)) {
            return false;
        }
        
        // Area filters
        if (filters.minArea && row.area_sqm && row.area_sqm < parseInt(filters.minArea)) {
            return false;
        }
        if (filters.maxArea && row.area_sqm && row.area_sqm > parseInt(filters.maxArea)) {
            return false;
        }
        
        // Boolean filters
        if (filters.isRental !== 'all' && row.is_rental !== (filters.isRental === 'true')) {
            return false;
        }
        if (filters.hasBalcony !== 'all' && row.has_balcony !== (filters.hasBalcony === 'true')) {
            return false;
        }
        if (filters.hasParking !== 'all' && row.has_parking !== (filters.hasParking === 'true')) {
            return false;
        }
        if (filters.hasGarden !== 'all' && row.has_garden !== (filters.hasGarden === 'true')) {
            return false;
        }
        if (filters.hasView !== 'all' && row.has_view !== (filters.hasView === 'true')) {
            return false;
        }
        if (filters.hasAirConditioning !== 'all' && row.has_air_conditioning !== (filters.hasAirConditioning === 'true')) {
            return false;
        }
        
        return true;
    });
    
    const updateFilter = (key, value) => {
        setFilters(prev => ({ ...prev, [key]: value }));
    };
    
    const clearFilters = () => {
        setFilters({
            city: '',
            region: '',
            minPrice: '',
            maxPrice: '',
            minRooms: '',
            maxRooms: '',
            minArea: '',
            maxArea: '',
            isRental: 'all',
            hasBalcony: 'all',
            hasParking: 'all',
            hasGarden: 'all',
            hasView: 'all',
            hasAirConditioning: 'all'
        });
    };
    
    return (
        <>
            <style>
                {`
                    @keyframes spin {
                        0% { transform: rotate(0deg); }
                        100% { transform: rotate(360deg); }
                    }
                    .explorer-table {
                        width: 100%;
                        border-collapse: collapse;
                        margin: 2rem 0;
                        background: rgba(0,0,0,0.6);
                        border-radius: 8px;
                        overflow: hidden;
                        font-size: 0.85rem;
                    }
                    .explorer-table th, .explorer-table td {
                        padding: 0.5rem 0.75rem;
                        border-bottom: 1px solid rgba(255,255,255,0.1);
                        white-space: nowrap;
                        overflow: hidden;
                        text-overflow: ellipsis;
                        max-width: 150px;
                    }
                    .explorer-table th {
                        background: rgba(255,255,255,0.08);
                        font-weight: 600;
                        position: sticky;
                        top: 0;
                    }
                    .explorer-table tr:last-child td {
                        border-bottom: none;
                    }
                    .pagination {
                        display: flex;
                        gap: 0.5rem;
                        justify-content: center;
                        align-items: center;
                        margin-bottom: 2rem;
                    }
                    .pagination button {
                        background: #fff;
                        color: #222;
                        border: none;
                        border-radius: 4px;
                        padding: 0.4rem 0.8rem;
                        cursor: pointer;
                        font-weight: 600;
                        transition: background 0.2s;
                    }
                    .pagination button:disabled {
                        background: #ccc;
                        color: #888;
                        cursor: not-allowed;
                    }
                    .pagination .active {
                        background: #0078d4;
                        color: #fff;
                    }
                    .data-count {
                        text-align: center;
                        margin: 1rem 0;
                        font-size: 1rem;
                        color: rgba(255, 255, 255, 0.8);
                    }
                    .filters-container {
                        background: rgba(0,0,0,0.7);
                        border-radius: 12px;
                        padding: 2rem;
                        margin: 2rem 0;
                        width: 100%;
                        max-width: 1200px;
                    }
                    .filters-grid {
                        display: grid;
                        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                        gap: 1rem;
                        margin-bottom: 1rem;
                    }
                    .filter-group {
                        display: flex;
                        flex-direction: column;
                        gap: 0.5rem;
                    }
                    .filter-label {
                        font-size: 0.9rem;
                        font-weight: 600;
                        color: rgba(255, 255, 255, 0.9);
                    }
                    .filter-input, .filter-select {
                        padding: 0.5rem;
                        border: 1px solid rgba(255, 255, 255, 0.3);
                        border-radius: 6px;
                        background: rgba(0, 0, 0, 0.6);
                        color: white;
                        font-size: 0.9rem;
                    }
                    .filter-select option {
                        background: rgba(0, 0, 0, 0.9);
                        color: white;
                    }
                    .filter-input::placeholder {
                        color: rgba(255, 255, 255, 0.6);
                    }
                    .filter-input:focus, .filter-select:focus {
                        outline: none;
                        border-color: #667eea;
                        background: rgba(0, 0, 0, 0.8);
                    }
                    .clear-filters-btn {
                        background: #ff6b6b;
                        color: white;
                        border: none;
                        border-radius: 6px;
                        padding: 0.6rem 1.2rem;
                        cursor: pointer;
                        font-weight: 600;
                        transition: background 0.2s;
                    }
                    .clear-filters-btn:hover {
                        background: #ff5252;
                    }
                `}
            </style>
            <div
                style={{
                    minHeight: "100vh",
                    maxHeight: "100vh",
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
                        Data Explorer
                    </h1>
                    <div className="data-count">
                        Showing {filteredData.length} of {dataArray.length} properties
                    </div>
                </main>
                
                {/* Filters */}
                <div className="filters-container">
                    <h3 style={{ marginBottom: "1.5rem", color: "white", textAlign: "center" }}>Filters</h3>
                    <div className="filters-grid">
                        <div className="filter-group">
                            <label className="filter-label">City</label>
                            <input
                                type="text"
                                className="filter-input"
                                placeholder="Enter city..."
                                value={filters.city}
                                onChange={(e) => updateFilter('city', e.target.value)}
                            />
                        </div>
                        
                        <div className="filter-group">
                            <label className="filter-label">Region</label>
                            <input
                                type="text"
                                className="filter-input"
                                placeholder="Enter region..."
                                value={filters.region}
                                onChange={(e) => updateFilter('region', e.target.value)}
                            />
                        </div>
                        
                        <div className="filter-group">
                            <label className="filter-label">Min Price (CHF)</label>
                            <input
                                type="number"
                                className="filter-input"
                                placeholder="Min price..."
                                value={filters.minPrice}
                                onChange={(e) => updateFilter('minPrice', e.target.value)}
                            />
                        </div>
                        
                        <div className="filter-group">
                            <label className="filter-label">Max Price (CHF)</label>
                            <input
                                type="number"
                                className="filter-input"
                                placeholder="Max price..."
                                value={filters.maxPrice}
                                onChange={(e) => updateFilter('maxPrice', e.target.value)}
                            />
                        </div>
                        
                        <div className="filter-group">
                            <label className="filter-label">Min Rooms</label>
                            <input
                                type="number"
                                step="0.5"
                                className="filter-input"
                                placeholder="Min rooms..."
                                value={filters.minRooms}
                                onChange={(e) => updateFilter('minRooms', e.target.value)}
                            />
                        </div>
                        
                        <div className="filter-group">
                            <label className="filter-label">Max Rooms</label>
                            <input
                                type="number"
                                step="0.5"
                                className="filter-input"
                                placeholder="Max rooms..."
                                value={filters.maxRooms}
                                onChange={(e) => updateFilter('maxRooms', e.target.value)}
                            />
                        </div>
                        
                        <div className="filter-group">
                            <label className="filter-label">Min Area (m²)</label>
                            <input
                                type="number"
                                className="filter-input"
                                placeholder="Min area..."
                                value={filters.minArea}
                                onChange={(e) => updateFilter('minArea', e.target.value)}
                            />
                        </div>
                        
                        <div className="filter-group">
                            <label className="filter-label">Max Area (m²)</label>
                            <input
                                type="number"
                                className="filter-input"
                                placeholder="Max area..."
                                value={filters.maxArea}
                                onChange={(e) => updateFilter('maxArea', e.target.value)}
                            />
                        </div>
                        
                        <div className="filter-group">
                            <label className="filter-label">Property Type</label>
                            <select
                                className="filter-select"
                                value={filters.isRental}
                                onChange={(e) => updateFilter('isRental', e.target.value)}
                            >
                                <option value="all">All</option>
                                <option value="true">Rental</option>
                                <option value="false">Purchase</option>
                            </select>
                        </div>
                        
                        <div className="filter-group">
                            <label className="filter-label">Balcony</label>
                            <select
                                className="filter-select"
                                value={filters.hasBalcony}
                                onChange={(e) => updateFilter('hasBalcony', e.target.value)}
                            >
                                <option value="all">All</option>
                                <option value="true">Yes</option>
                                <option value="false">No</option>
                            </select>
                        </div>
                        
                        <div className="filter-group">
                            <label className="filter-label">Parking</label>
                            <select
                                className="filter-select"
                                value={filters.hasParking}
                                onChange={(e) => updateFilter('hasParking', e.target.value)}
                            >
                                <option value="all">All</option>
                                <option value="true">Yes</option>
                                <option value="false">No</option>
                            </select>
                        </div>
                        
                        <div className="filter-group">
                            <label className="filter-label">Garden</label>
                            <select
                                className="filter-select"
                                value={filters.hasGarden}
                                onChange={(e) => updateFilter('hasGarden', e.target.value)}
                            >
                                <option value="all">All</option>
                                <option value="true">Yes</option>
                                <option value="false">No</option>
                            </select>
                        </div>
                    </div>
                    <div style={{ textAlign: "center", marginTop: "1rem" }}>
                        <button className="clear-filters-btn" onClick={clearFilters}>
                            Clear All Filters
                        </button>
                    </div>
                </div>
                
                {/* Table Component */}
                <div style={{ 
                    width: "100%", 
                    maxWidth: "1200px", 
                    overflowX: "auto",
                    overflowY: "auto",
                    maxHeight: "70vh",
                    borderRadius: "8px"
                }}>
                    <table className="explorer-table">
                        <thead>
                            <tr>
                                <th>ID</th>
                                <th>Title</th>
                                <th>Address</th>
                                <th>ZIP</th>
                                <th>City</th>
                                <th>Region</th>
                                <th>Price</th>
                                <th>Rooms</th>
                                <th>Area (m²)</th>
                                <th>Floor</th>
                                <th>Available</th>
                                <th>Balcony</th>
                                <th>Rental</th>
                                <th>New</th>
                                <th>View</th>
                                <th>Garden</th>
                                <th>Parking</th>
                                <th>AC</th>
                            </tr>
                        </thead>
                        <tbody>
                            {filteredData.map((row, index) => (
                                <tr key={row.id || index}>
                                    <td>{row.id}</td>
                                    <td>{row.title}</td>
                                    <td>{row.address}</td>
                                    <td>{row.zip_code}</td>
                                    <td>{row.city}</td>
                                    <td>{row.region}</td>
                                    <td>{row.price ? `CHF ${row.price.toLocaleString()}` : '-'}</td>
                                    <td>{row.rooms}</td>
                                    <td>{row.area_sqm}</td>
                                    <td>{row.floor}</td>
                                    <td>{row.availability_date ? new Date(row.availability_date).toLocaleDateString() : '-'}</td>
                                    <td>{row.has_balcony ? '✓' : '✗'}</td>
                                    <td>{row.is_rental ? '✓' : '✗'}</td>
                                    <td>{row.is_new ? '✓' : '✗'}</td>
                                    <td>{row.has_view ? '✓' : '✗'}</td>
                                    <td>{row.has_garden ? '✓' : '✗'}</td>
                                    <td>{row.has_parking ? '✓' : '✗'}</td>
                                    <td>{row.has_air_conditioning ? '✓' : '✗'}</td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>

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
};