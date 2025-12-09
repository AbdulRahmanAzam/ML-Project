import { useState } from 'react'
import './App.css'

const API_URL = 'http://139.59.23.227/api'

function App() {
  const [formData, setFormData] = useState({
    title: '',
    description: '',
    company_profile: '',
    requirements: '',
    benefits: '',
    location: '',
    has_company_logo: 1
  })

  const [prediction, setPrediction] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const handleChange = (e) => {
    const { name, value, type, checked } = e.target
    setFormData(prev => ({
      ...prev,
      [name]: type === 'checkbox' ? (checked ? 1 : 0) : value
    }))
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    setLoading(true)
    setError(null)
    setPrediction(null)

    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData)
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || 'Failed to get prediction')
      }

      const data = await response.json()
      setPrediction(data)
    } catch (err) {
      setError(err.message || 'An error occurred while making the prediction')
    } finally {
      setLoading(false)
    }
  }

  const resetForm = () => {
    setFormData({
      title: '',
      description: '',
      company_profile: '',
      requirements: '',
      benefits: '',
      location: '',
      has_company_logo: 1
    })
    setPrediction(null)
    setError(null)
  }

  return (
    <div className="app">
      {/* Header */}
      <header className="header">
        <div className="header-content">
          <div className="logo">
            <svg className="logo-icon" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path d="M12 2L2 7L12 12L22 7L12 2Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              <path d="M2 17L12 22L22 17" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              <path d="M2 12L12 17L22 12" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
            <span>JobGuard AI</span>
          </div>
          <nav className="nav-links">
            <a href="#how-it-works">How It Works</a>
            <a href="#about">About</a>
          </nav>
        </div>
      </header>

      {/* Hero Section */}
      <section className="hero">
        <h1>Fake Job Posting Detector</h1>
        <p>Protect yourself from fraudulent job postings using our advanced AI-powered detection system</p>
      </section>

      {/* Main Content */}
      <main className="main-content">
        <div className="container">
          {/* Form Section */}
          <div className="form-container">
            <section className="form-section">
            <div className="section-header">
              <h2>Analyze Job Posting</h2>
              <p>Fill in the job details below to check if it's legitimate or potentially fraudulent</p>
            </div>

            <form onSubmit={handleSubmit} className="job-form">
              {/* Job Title */}
              <div className="form-group">
                <label htmlFor="title">
                  Job Title <span className="required">*</span>
                </label>
                <input
                  type="text"
                  id="title"
                  name="title"
                  value={formData.title}
                  onChange={handleChange}
                  placeholder="e.g., Senior Software Engineer"
                  required
                />
              </div>

              {/* Location */}
              <div className="form-group">
                <label htmlFor="location">
                  Location <span className="required">*</span>
                </label>
                <input
                  type="text"
                  id="location"
                  name="location"
                  value={formData.location}
                  onChange={handleChange}
                  placeholder="e.g., New York, NY, USA"
                  required
                />
              </div>

              {/* Company Logo */}
              <div className="form-group checkbox-group">
                <label className="checkbox-label">
                  <input
                    type="checkbox"
                    name="has_company_logo"
                    checked={formData.has_company_logo === 1}
                    onChange={handleChange}
                  />
                  <span className="checkmark"></span>
                  Company has a logo
                </label>
              </div>

              {/* Company Profile */}
              <div className="form-group">
                <label htmlFor="company_profile">
                  Company Profile <span className="required">*</span>
                </label>
                <textarea
                  id="company_profile"
                  name="company_profile"
                  value={formData.company_profile}
                  onChange={handleChange}
                  placeholder="Describe the company, its mission, and what it does..."
                  rows="4"
                  required
                />
              </div>

              {/* Job Description */}
              <div className="form-group">
                <label htmlFor="description">
                  Job Description <span className="required">*</span>
                </label>
                <textarea
                  id="description"
                  name="description"
                  value={formData.description}
                  onChange={handleChange}
                  placeholder="Describe the role, responsibilities, and what the job entails..."
                  rows="5"
                  required
                />
              </div>

              {/* Requirements */}
              <div className="form-group">
                <label htmlFor="requirements">
                  Requirements <span className="required">*</span>
                </label>
                <textarea
                  id="requirements"
                  name="requirements"
                  value={formData.requirements}
                  onChange={handleChange}
                  placeholder="List the required skills, qualifications, and experience..."
                  rows="4"
                  required
                />
              </div>

              {/* Benefits */}
              <div className="form-group">
                <label htmlFor="benefits">
                  Benefits <span className="required">*</span>
                </label>
                <textarea
                  id="benefits"
                  name="benefits"
                  value={formData.benefits}
                  onChange={handleChange}
                  placeholder="List the benefits offered (health insurance, PTO, etc.)..."
                  rows="3"
                  required
                />
              </div>

              {/* Submit Button */}
              <div className="form-actions">
                <button type="submit" className="btn btn-primary" disabled={loading}>
                  {loading ? (
                    <>
                      <span className="spinner"></span>
                      Analyzing...
                    </>
                  ) : (
                    <>
                      <svg className="btn-icon" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path d="M21 21L15 15M17 10C17 13.866 13.866 17 10 17C6.13401 17 3 13.866 3 10C3 6.13401 6.13401 3 10 3C13.866 3 17 6.13401 17 10Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                      </svg>
                      Analyze Job Posting
                    </>
                  )}
                </button>
                <button type="button" className="btn btn-secondary" onClick={resetForm}>
                  Clear Form
                </button>
              </div>
            </form>
          </section>
          </div>
        </div>
      </main>

      {/* Modal for Results */}
      {(prediction || error) && (
        <>
          <div className="modal-overlay" onClick={() => { setPrediction(null); setError(null); }}></div>
          <div className="modal">
            <button className="modal-close" onClick={() => { setPrediction(null); setError(null); }}>
              <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M18 6L6 18M6 6L18 18" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              </svg>
            </button>

            <div className="modal-content">
              {error && (
                <div className="error-card">
                  <div className="error-icon">⚠️</div>
                  <div className="error-content">
                    <h3>Error</h3>
                    <p>{error}</p>
                  </div>
                </div>
              )}

              {prediction && (
                <>
                  {/* Main Result Card */}
                  <div className={`result-card-modal ${prediction.is_fake ? 'fake' : 'real'}`}>
                    <div className="result-icon-large">
                      {prediction.is_fake ? (
                        <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                          <path d="M12 9V11M12 15H12.01M5.07183 19H18.9282C20.4678 19 21.4301 17.3333 20.6603 16L13.7321 4C12.9623 2.66667 11.0378 2.66667 10.268 4L3.33978 16C2.56998 17.3333 3.53223 19 5.07183 19Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                        </svg>
                      ) : (
                        <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                          <path d="M9 12L11 14L15 10M21 12C21 16.9706 16.9706 21 12 21C7.02944 21 3 16.9706 3 12C3 7.02944 7.02944 3 12 3C16.9706 3 21 7.02944 21 12Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                        </svg>
                      )}
                    </div>
                    <h2 className="result-title">{prediction.is_fake ? 'Potentially Fraudulent' : 'Likely Legitimate'}</h2>
                    <p className="result-message">{prediction.message}</p>
                    
                    {/* Confidence Level Badge */}
                    <div className="confidence-level-badge">
                      <span className="confidence-label">Confidence Level</span>
                      <span className="confidence-value">{prediction.confidence_level}</span>
                    </div>
                  </div>
                </>  
              )}
            </div>
          </div>
        </>
      )}

      {/* How It Works Section */}
      <section id="how-it-works" className="how-it-works">
        <div className="container">
          <h2>How It Works</h2>
          <div className="steps-grid">
            <div className="step-card">
              <div className="step-number">1</div>
              <h3>Enter Job Details</h3>
              <p>Fill in the job posting information including title, description, requirements, and company details.</p>
            </div>
            <div className="step-card">
              <div className="step-number">2</div>
              <h3>AI Analysis</h3>
              <p>Our machine learning model analyzes the text patterns, language, and indicators associated with fraudulent postings.</p>
            </div>
            <div className="step-card">
              <div className="step-number">3</div>
              <h3>Get Results</h3>
              <p>Receive an instant assessment with probability scores and confidence levels to help you make informed decisions.</p>
            </div>
          </div>
        </div>
      </section>

      {/* About Section */}
      <section id="about" className="about">
        <div className="container">
          <h2>About JobGuard AI</h2>
          <p>
            JobGuard AI uses advanced machine learning techniques trained on thousands of job postings 
            to identify potential fraudulent listings. Our model analyzes multiple features including 
            job descriptions, company profiles, requirements, and benefits to detect suspicious patterns 
            commonly found in fake job postings.
          </p>
          <div className="features-grid">
            <div className="feature">
              <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M9 12L11 14L15 10M12 3L13.9101 4.87147L16.5 4.20577L17.2184 6.78155L19.7942 7.5L19.1285 10.0899L21 12L19.1285 13.9101L19.7942 16.5L17.2184 17.2184L16.5 19.7942L13.9101 19.1285L12 21L10.0899 19.1285L7.5 19.7942L6.78155 17.2184L4.20577 16.5L4.87147 13.9101L3 12L4.87147 10.0899L4.20577 7.5L6.78155 6.78155L7.5 4.20577L10.0899 4.87147L12 3Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              </svg>
              <h4>High Accuracy</h4>
              <p>Trained on real-world data with proven detection rates</p>
            </div>
            <div className="feature">
              <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M13 10V3L4 14H11V21L20 10H13Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              </svg>
              <h4>Instant Results</h4>
              <p>Get predictions in seconds with detailed confidence scores</p>
            </div>
            <div className="feature">
              <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M12 22C17.5228 22 22 17.5228 22 12C22 6.47715 17.5228 2 12 2C6.47715 2 2 6.47715 2 12C2 17.5228 6.47715 22 12 22Z" stroke="currentColor" strokeWidth="2"/>
                <path d="M12 16V12M12 8H12.01" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
              </svg>
              <h4>Stay Informed</h4>
              <p>Protect yourself from scams and make better career decisions</p>
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="footer">
        <div className="container">
          <p>© 2025 JobGuard AI - Fake Job Detection System</p>
          <p className="disclaimer">
            This tool provides predictions based on machine learning analysis. 
            Always conduct your own research before applying to any job.
          </p>
        </div>
      </footer>
    </div>
  )
}

export default App
