import React from 'react';

export default function Explorer() {


    // Dummy data for demonstration
    // Fetch data from backend API
    const [data, setData] = React.useState([]);
    React.useEffect(() => {
        fetch("http://localhost:5000/api/properties")
            .then(res => res.json())
            .then(setData)
            .catch(() => setData([]));
    }, []);

    const [page, setPage] = React.useState(1);
    const rowsPerPage = 5;
    const pageCount = Math.ceil(data.length / rowsPerPage);

    const paginatedData = data.slice((page - 1) * rowsPerPage, page * rowsPerPage);

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
                        Data Explorer
                    </h1>
                </main>
                
                {/* Table Component */}
                <div style={{ 
                    width: "100%", 
                    maxWidth: "1200px", 
                    overflowX: "auto",
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
                            {paginatedData.map((row) => (
                                <tr key={row.id}>
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
                    <div className="pagination">
                        <button onClick={() => setPage(page - 1)} disabled={page === 1}>
                            Prev
                        </button>
                        {Array.from({ length: pageCount }, (_, i) => (
                            <button
                                key={i + 1}
                                className={page === i + 1 ? "active" : ""}
                                onClick={() => setPage(i + 1)}
                            >
                                {i + 1}
                            </button>
                        ))}
                        <button onClick={() => setPage(page + 1)} disabled={page === pageCount}>
                            Next
                        </button>
                    </div>
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