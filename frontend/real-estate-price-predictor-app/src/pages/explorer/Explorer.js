import React from 'react';

export default function Explorer() {


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
                        Data Explorer
                    </h1>
                </main>
                
                {/* This is where the table should go */}


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