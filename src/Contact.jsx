import React, { useRef } from "react";
import "./Contact.css";
import emailjs from "@emailjs/browser";

const Contact = () => {
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
    <div>
      <div className="Contact">
        <h1>Get in Touch</h1>

        <div className="Contact-info">
          <p>
            📧 Email:
            <a href="mailto:deepscan460@gmail.com">deepscan460@gmail.com</a>
          </p>
        </div>

        <form ref={form} onSubmit={sendEmail} className="Contact-form_info">
          <div className="Contact-left">
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
          </div>

          <div className="Contact-right">
            <textarea name="message" placeholder="Your Message *"></textarea>
          </div>

          <input
            className="Contact-submit"
            type="submit"
            value="Send Message"
          />
        </form>
      </div>
    </div>
  );
};

export default Contact;
