import React, { useState, useEffect } from "react";
import axios from "axios";
import Welcome from "./Welcome";
import { useNavigate, Link } from "react-router-dom"
function SignIn() {
  const [formData, setFormData] = useState({
    email: "",
    password: "", 
   
  });

  const { email, password } = formData;
  const history=useNavigate();
  const onChange = (e) =>
    setFormData({ ...formData, [e.target.name]: e.target.value });
  
  const onSubmit = async (e) => {
    e.preventDefault();

    const user = {
      email,
      password,
    };
    e.preventDefault();

    try{

        await axios.post("http://localhost:5000/api/login",{
            email,password
        })
        .then(res=>{ 
            if(res.data.status=="exist"){
                history("../Welcome",{state:{username:res.data.username}})
                console.log(res.data.username)
            }
            else if(res.data=="pnotexist"){
                alert("Incorrect Password")
            } 
            else 
            {
              alert("User is not sign Up")
            }
        })
        .catch(e=>{
            alert("wrong details")
            console.log(e);
        })

    }
    catch(e){
        console.log(e);

    }
    // fetch("http://localhost:5000/api/login", {
    //   method: "POST",
    //   headers: {
    //     "Content-Type": "application/json",
    //   },
    //   body: JSON.stringify(formData),
    // })
      // .then((response) => response.json())
      // .then((data) => {
      //   console.log(data);
      //   navigate("../welcome");
      // })
      // .catch((error) => {
      //   console.error("Hello Mohsina");
      // });
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
      <div className="container mt-5 ">
        <form onSubmit={onSubmit}>
          <div className="mb-3">
            <div className="form-group">
              <label htmlFor="exampleInputEmail1">Email address</label>
              <input
                type="email"
                className="form-control"
                id="exampleInputEmail1"
                aria-describedby="emailHelp"
                value={email}
                onChange={onChange}
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
                value={password}
                onChange={onChange}
              />
            </div>
          </div>
          <button type="submit" className="btn btn-success">
            Submit
          </button>
        </form>
      </div>
    </>
  );
}
export default SignIn;
