import React from "react";

export default function AboutPage() {
  return (
    <div
      style={{
        minHeight: "100vh",
        background: `
          linear-gradient(
            rgba(0, 0, 0, 0.4),
            rgba(0, 0, 0, 0.7)
          ),
          url('https://avantecture.com/wp-content/uploads/2021/10/Bruderhaus-Nr-2-aussen-13.jpg')
        `,
        backgroundSize: "cover",
        backgroundPosition: "center",
        backgroundRepeat: "no-repeat",
        backgroundAttachment: "fixed",
        fontFamily: '"Helvetica Neue", Helvetica, Arial, sans-serif',
        color: "white",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        textAlign: "center",
        padding: "2rem"
      }}
    >
      <main
        style={{
          width: "100%",
          maxWidth: "900px",
          borderRadius: "20px",
          padding: "4rem",
          color: "white"
        }}
      >
        <div style={{ textAlign: "center", marginBottom: "4rem" }}>
          <h1 style={{ 
            fontSize: "3.5rem", 
            marginBottom: "1rem",
            fontWeight: "700",
            color: "white"
          }}>
            🏠 Real Estate Price Predictor
          </h1>
          <div style={{
            width: "100px",
            height: "4px",
            background: "linear-gradient(90deg, #667eea, #764ba2)",
            margin: "0 auto",
            borderRadius: "2px"
          }}></div>
        </div>

        <div style={{
          display: "grid",
          gridTemplateColumns: "1fr 300px",
          gap: "3rem",
          alignItems: "stretch",
          marginBottom: "4rem"
        }}>
          <div style={{
            background: "rgba(255, 255, 255, 0.9)",
            borderRadius: "15px",
            padding: "2rem",
            backdropFilter: "blur(10px)",
            display: "flex",
            alignItems: "center"
          }}>
            <p style={{ 
              fontSize: "1.3rem", 
              lineHeight: "1.8", 
              color: "#333",
              margin: "0"
            }}>
              Welcome to our <strong>AI-powered</strong> Real Estate Price Predictor! 🚀 
              A smart tool that helps you estimate rental or purchase prices for 
              residential properties in Switzerland. Whether you're hunting for your 
              next home or selling one, get instant, data-backed price estimates.
            </p>
          </div>
          <div style={{
            background: "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
            borderRadius: "15px",
            padding: "2rem",
            textAlign: "center",
            color: "white",
            display: "flex",
            flexDirection: "column",
            justifyContent: "center"
          }}>
            <div style={{ fontSize: "3rem", marginBottom: "1rem" }}>🎯</div>
            <h3 style={{ margin: "0 0 0.5rem 0", fontSize: "1.2rem" }}>Accurate Predictions</h3>
            <p style={{ margin: "0", fontSize: "0.9rem", opacity: "0.9" }}>
              Powered by ML algorithms & real Swiss property data
            </p>
          </div>
        </div>

        <div style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))",
          gap: "2rem",
          marginBottom: "4rem"
        }}>
          <div style={{
            background: "linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%)",
            borderRadius: "15px",
            padding: "2rem",
            color: "white",
            textAlign: "center"
          }}>
            <div style={{ fontSize: "3rem", marginBottom: "1rem" }}>🎯</div>
            <h3 style={{ fontSize: "1.5rem", marginBottom: "1rem", fontWeight: "600" }}>Why we built this</h3>
            <p style={{ fontSize: "1rem", lineHeight: "1.6", margin: "0", opacity: "0.95" }}>
              Property hunting is overwhelming! Prices vary wildly and it's hard to know what's fair. 
              We use real data and ML to reduce guesswork and promote transparency.
            </p>
          </div>

          <div style={{
            background: "linear-gradient(135deg, #4ecdc4 0%, #2fb3a8 100%)",
            borderRadius: "15px",
            padding: "2rem",
            color: "white",
            textAlign: "center"
          }}>
            <div style={{ fontSize: "3rem", marginBottom: "1rem" }}>🔧</div>
            <h3 style={{ fontSize: "1.5rem", marginBottom: "1rem", fontWeight: "600" }}>What we used</h3>
            <ul style={{
              listStyle: "none",
              padding: "0",
              margin: "0",
              fontSize: "0.95rem",
              lineHeight: "1.6"
            }}>
              <li style={{ marginBottom: "0.5rem" }}>📊 Flatfox & UrbanHome data</li>
              <li style={{ marginBottom: "0.5rem" }}>🤖 Random Forest & XGBoost</li>
              <li style={{ marginBottom: "0.5rem" }}>🖼️ Image feature extraction</li>
              <li style={{ marginBottom: "0.5rem" }}>📍 Location clustering</li>
              <li>⚡ Real-time predictions</li>
            </ul>
          </div>
        </div>

        <div style={{
          display: "grid",
          gridTemplateColumns: "400px 1fr",
          gap: "3rem",
          alignItems: "stretch",
          marginBottom: "4rem"
        }}>
          <div style={{
            background: "linear-gradient(135deg, #a8edea 0%, #fed6e3 100%)",
            borderRadius: "15px",
            padding: "2rem",
            textAlign: "center",
            display: "flex",
            flexDirection: "column",
            justifyContent: "center"
          }}>
            <div style={{ fontSize: "4rem", marginBottom: "1rem" }}>📈</div>
            <h3 style={{ color: "white", fontSize: "1.3rem", margin: "0" }}>Smart Analytics</h3>
            <p style={{ color: "#ddd", fontSize: "0.9rem", margin: "0.5rem 0 0 0" }}>
              Size, location, rooms, images - we analyze it all!
            </p>
          </div>
          <div style={{
            background: "rgba(255, 255, 255, 0.9)",
            borderRadius: "15px",
            padding: "2rem",
            backdropFilter: "blur(10px)",
            display: "flex",
            flexDirection: "column",
            justifyContent: "center"
          }}>
            <h2 style={{ 
              fontSize: "2.2rem", 
              marginBottom: "1.5rem",
              color: "#333",
              fontWeight: "600"
            }}>✅ What it can do</h2>
            <p style={{ 
              fontSize: "1.2rem", 
              lineHeight: "1.7", 
              color: "#333",
              margin: "0"
            }}>
              Our tool analyzes property features like size, location, number of rooms,
              and even <strong>image content</strong> to give instant price estimates. 
              It supports both rentals and purchases. Explore the map, check visual stats,
              and get predictions with just a few clicks! 🎉
            </p>
          </div>
        </div>

        <div style={{
          background: "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
          borderRadius: "20px",
          padding: "3rem",
          color: "white",
          marginBottom: "3rem"
        }}>
          <h2 style={{ 
            fontSize: "2.5rem", 
            marginBottom: "2rem",
            fontWeight: "600",
            textAlign: "center"
          }}>👥 Meet Our Amazing Team</h2>
          
          <div style={{
            display: "grid",
            gridTemplateColumns: "1fr 1fr",
            gap: "2rem"
          }}>
            <div style={{
              background: "rgba(255, 255, 255, 0.15)",
              borderRadius: "15px",
              padding: "2rem",
              textAlign: "center",
              backdropFilter: "blur(10px)"
            }}>
              <div style={{ fontSize: "4rem", marginBottom: "1rem" }}>👨‍💻</div>
              <h3 style={{ fontSize: "1.3rem", marginBottom: "0.5rem", fontWeight: "600" }}>
                Antonio Michal Verdile
              </h3>
              <p style={{ fontSize: "1rem", opacity: "0.9", margin: "0" }}>
                Full Stack Dev · UI/UX · ML integration
              </p>
            </div>
            
            <div style={{
              background: "rgba(255, 255, 255, 0.15)",
              borderRadius: "15px",
              padding: "2rem",
              textAlign: "center",
              backdropFilter: "blur(10px)"
            }}>
              <div style={{ fontSize: "4rem", marginBottom: "1rem" }}>🧠</div>
              <h3 style={{ fontSize: "1.3rem", marginBottom: "0.5rem", fontWeight: "600" }}>
                Alessandro Mazzeo
              </h3>
              <p style={{ fontSize: "1rem", opacity: "0.9", margin: "0" }}>
                ML Engineer · Feature engineering · Evaluation
              </p>
            </div>
            
            <div style={{
              background: "rgba(255, 255, 255, 0.15)",
              borderRadius: "15px",
              padding: "2rem",
              textAlign: "center",
              backdropFilter: "blur(10px)"
            }}>
              <div style={{ fontSize: "4rem", marginBottom: "1rem" }}>🗃️</div>
              <h3 style={{ fontSize: "1.3rem", marginBottom: "0.5rem", fontWeight: "600" }}>
                Josh Richt
              </h3>
              <p style={{ fontSize: "1rem", opacity: "0.9", margin: "0" }}>
                Data Engineer · Web scraping · DB design
              </p>
            </div>

            <div style={{
              background: "rgba(255, 255, 255, 0.15)",
              borderRadius: "15px",
              padding: "2rem",
              textAlign: "center",
              backdropFilter: "blur(10px)"
            }}>
              <div style={{ fontSize: "4rem", marginBottom: "1rem" }}>📊</div>
              <h3 style={{ fontSize: "1.3rem", marginBottom: "0.5rem", fontWeight: "600" }}>
                Enmanuel Lizardo
              </h3>
              <p style={{ fontSize: "1rem", opacity: "0.9", margin: "0" }}>
                Data Scientist · Analytics · Research
              </p>
            </div>
          </div>
        </div>

        <div style={{
          textAlign: "center",
          padding: "2rem",
          background: "linear-gradient(135deg, #ffeaa7 0%, #fab1a0 100%)",
          borderRadius: "15px",
          color: "white"
        }}>
          <div style={{ fontSize: "2rem", marginBottom: "1rem" }}>☕</div>
          <p style={{ 
            fontSize: "1.2rem", 
            fontWeight: "500",
            margin: "0"
          }}>
            Built with ❤️ and lots of Python, JavaScript, and coffee!
          </p>
        </div>
      </main>

      <footer
        style={{
          textAlign: "center",
          padding: "1rem",
          fontSize: "0.85rem",
          opacity: 0.7
        }}
      >
        Michael · Josh · Enmanuel · Alessandro
      </footer>
    </div>
  );
}
