import "./App.css";

function App() {
  return (
    <div className="app">
      <main className="login-container">

        {/* Brand Section */}
        <section className="brand-section">

          <div className="logo">
            <span className="logo-eye">◉</span>
          </div>

          <h1>SeeBeyond</h1>

          <p className="tagline">
            AI-assisted retinal screening
          </p>

          <p className="description">
            Understand your retinal health with AI-powered
            diabetic retinopathy screening and retinal
            lesion localization.
          </p>

          <div className="feature-list">

            <div className="feature">
              <span>✓</span>
              <p>AI-powered DR screening</p>
            </div>

            <div className="feature">
              <span>✓</span>
              <p>Retinal lesion localization</p>
            </div>

            <div className="feature">
              <span>✓</span>
              <p>Personal screening history</p>
            </div>

          </div>

        </section>


        {/* Login Card */}
        <section className="login-card">

          <div className="card-header">

            <h2>Welcome to SeeBeyond</h2>

            <p>
              Sign in to access your AI screening
              dashboard and history.
            </p>

          </div>


          {/* Google Login */}
          <button
            className="google-button"
            type="button"
          >
            <span className="google-icon">G</span>

            <span>
              Continue with Google
            </span>

          </button>


          {/* Divider */}
          <div className="divider">
            <span>Secure access</span>
          </div>


          {/* Terms */}
          <p className="terms">
            By continuing, you agree to use SeeBeyond
            as an AI-assisted screening and research tool.
          </p>


          {/* Medical Disclaimer */}
          <p className="medical-note">
            SeeBeyond does not replace professional eye care
            or clinical diagnosis.
          </p>

        </section>

      </main>


      {/* Footer */}
      <footer>

        <span>
          SeeBeyond
        </span>

        <span>
          AI-assisted retinal screening
        </span>

      </footer>

    </div>
  );
}

export default App;
