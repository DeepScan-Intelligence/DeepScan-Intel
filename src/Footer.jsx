import React, { useRef } from "react";
import "./Footer.css";
import emailjs from "@emailjs/browser";

const Footer = () => {
  const form = useRef();

  const sendEmail = (e) => {
    e.preventDefault();
    const formData = new FormData(form.current);
    const name = formData.get("name");
    const email = formData.get("email");
    const message = formData.get("message");

    if (!email || !message || !name) {
      alert("Please fill in all the required fields.");
      return;
    }

    emailjs
      .sendForm("service_s6pqsvm", "template_cweek1b", form.current, {
        publicKey: "2m0RmHkTpHCN3XgIq",
      })
      .then(
        () => {
          alert("Message sent successfully!");
        },
        (error) => {
          alert("Message failed to send.");
        }
      );
  };

  return (
    <footer className="Footer">
      <h1>Get in Touch</h1>

      <div className="Contact-info">
        <p>
          📧 Email:{" "}
          <a href="mailto:deepscan460@gmail.com">deepscan460@gmail.com</a>
        </p>
      </div>

      <form ref={form} onSubmit={sendEmail} className="Contact-form_info">
        <input
          className="Contact-input"
          type="text"
          name="name"
          placeholder="Your Name *"
        />
        <input
          className="Contact-input"
          type="email"
          name="email"
          placeholder="Your Email *"
        />
        <textarea name="message" placeholder="Your Message *"></textarea>
        <input className="Contact-submit" type="submit" value="Send Message" />
      </form>

      <div className="Footer-copyright">
        © 2025 DeepScan Intelligence. All rights reserved.
      </div>
    </footer>
  );
};

export default Footer;
