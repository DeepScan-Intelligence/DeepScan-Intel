import React from "react";
import "./Homepage.css";
import Footer from "./Footer";
const Homepage = () => {
  return (
    <div className="Homepage">
      <div className="hero_image">
        <div className="hero-text">
          <img className="title" src="./Deepscan_title_start.png"></img>
          <button>SCAN NOW</button>
        </div>
      </div>

      <section className="what-we-do">
        <div className="what-we-do-image">
          <img src="./what we do.jpeg"></img>
        </div>
        <div className="what-we-do-content">
          <h2>What We Do</h2>
          <p className="what-we-do-text">
            At DeepScan Intelligence, we provide a simple yet powerful tool to
            identify manipulated videos. Our advanced AI technology scans videos
            for signs of digital alteration, helping you determine whether a
            video is genuine or a deepfake.
          </p>
        </div>
      </section>

      <section className="scanner-section">
        <h2>How Our Scanner Works</h2>
        <p>
          Our AI-powered scanner examines multiple aspects of a video to
          determine whether it’s real or a deepfake. From facial recognition to
          motion inconsistencies, here's how our system works:
        </p>
        <div className="scanner-grid">
          <div className="scanner-card">
            <img src="./facial analysis.jpeg" alt="Facial Analysis"></img>
            <h3>FACIAL ANALYSIS</h3>
            <p>
              The system scans facial movements and compares them against
              real-life data for signs of manipulation.
            </p>
          </div>
          <div className="scanner-card">
            <img src="./audio visual sync.jpeg" alt="Audio-Visual Sync"></img>
            <h3>AUDIO-VISUAL SYNC</h3>
            <p>
              It checks for any mismatches between the voice and lip movements.
            </p>
          </div>
          <div className="scanner-card">
            <img src="./lightings & shadows.jpg" alt="Audio-Visual Sync"></img>
            <h3>LIGHTING & SHADOWS</h3>
            <p>
              Inconsistencies in lighting and shadows are often a telltale sign
              of deepfakes.
            </p>
          </div>
          <div className="scanner-card">
            <img
              src="./behavioral patterns.jpeg"
              alt="Behavioral Patterns"
            ></img>

            <h3>BEHAVIORAL PATTERNS</h3>
            <p>
              The tool looks for irregularities in the video’s flow, like
              unnatural gestures or sudden pauses.
            </p>
          </div>
        </div>
      </section>

      <section className="why-we-started">
        <div className="why-we-started-image">
          <img
            src="./why we started this.jpeg"
            alt="Why We Started DeepScan"
          ></img>
        </div>
        <div className="why-we-started-content">
          <h2>Why We Started This</h2>
          <p className="why-we-started-text">
            Misinformation spreads quickly through altered media, and deepfakes
            make it even harder to trust what we see online. Our mission is to
            provide accessible, user-friendly tools that help identify and
            protect against deepfake content. We aim to raise awareness about
            the dangers of deepfakes and promote transparency in the digital
            age.
          </p>
        </div>
      </section>
      <section className="satisfied-clients">
        <h2>Satisfied Clients</h2>
        <div className="testimonial-grid">
          <div className="testimonial-card">
            <img className="testimonial-img" src="billg.png" alt="Bill Gates" />
            <h3>Bill Gates</h3>
            <p>
              "DeepScan Intelligence is an absolute game-changer. The AI-powered
              detection is incredibly accurate, helping to ensure that
              misinformation doesn’t spread. I highly recommend it for anyone
              concerned about digital authenticity!"
            </p>
            <div className="star-rating">★★★★★</div>
          </div>

          <div className="testimonial-card">
            <img className="testimonial-img" src="elonm.jpg" alt="Elon Musk" />
            <h3>Elon Musk</h3>
            <p>
              "Innovation is key, and DeepScan Intelligence is at the forefront
              of AI-driven video verification. The speed and precision of this
              tool are mind-blowing. This is the future of deepfake detection!"
            </p>
            <div className="star-rating">★★★★★</div>
          </div>

          <div className="testimonial-card">
            <img
              className="testimonial-img"
              src="markz.png"
              alt="Mark Zuckerberg"
            />
            <h3>Mark Zuckerberg</h3>
            <p>
              "As someone who values technology and security, I can confidently
              say that DeepScan Intelligence is one of the best tools for
              detecting manipulated media. The AI is incredibly smart, and the
              insights it provides are invaluable."
            </p>
            <div className="star-rating">★★★★★</div>
          </div>
        </div>
      </section>
      <Footer />
    </div>
  );
};

export default Homepage;
