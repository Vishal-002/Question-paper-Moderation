import React, { useState, useEffect } from "react";
import { Link } from "react-router-dom";
import axios from "axios";
import SignIn from "./SignIn";
import {  useNavigate } from "react-router-dom"; 

function SignUp() {
  const [formData, setFormData] = useState({
    username: "",
    email: "",
    password: "",
  });
  const navigate = useNavigate();
  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData({
      ...formData,
      [name]: value,
    });
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    fetch("http://localhost:5000/api/register", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(formData),
    })
      .then((response) => response.json())
      .then((data) => {
        console.log(data);
        navigate('../signin');
      })
      .catch((error) => {
        console.error(error);
      });
  };
  return (
    <>
      <header>
        <nav className="navbar">
          <div className="logo">Q_Moderation</div>
          <ul className="nav-links">
            <li><a href="signin">Login</a></li>
            <li><a href="signup">SignUp</a></li>
            <li><a href="/contact">Question Paper</a></li>
          </ul>
        </nav>
      </header>
      <div className="container mt-5">
        <form onSubmit={handleSubmit}>
          <div className="mb-3">
            <label htmlFor="name" className="form-label">
              Name
            </label>
            <input
              type="text"
              className="form-control"
              name="username"
              value={formData.username}
              onChange={handleChange}
            />
          </div>
          <div className="mb-3">
            <div className="form-group">
              <label htmlFor="exampleInputEmail1">Email address</label>
              <input
                type="email"
                className="form-control"
                id="exampleInputEmail1"
                aria-describedby="emailHelp"
                value={formData.email}
                onChange={handleChange}
                name="email"
              />
            </div>
          </div>
          <div className="mb-3">
            <div className="form-group">
              <label htmlFor="exampleInputPassword1">Password</label>
              <input
                type="text"
                className="form-control"
                id="exampleInputPassword1"
                name="password"
                value={formData.password}
                onChange={handleChange}
              />
            </div>
          </div>
          <div className="mb-3">
            <label htmlFor="name" className="form-label">
              Address
            </label>
            <input type="text" className="form-control" name="geolocation" />
          </div>
          <button type="submit" className="btn btn-success">
            Submit
          </button>
          <Link to="/login" className="m-3 btn btn-danger">
            Already a User
          </Link>
        </form>
      </div>
    </>
  );
}
export default SignUp;
