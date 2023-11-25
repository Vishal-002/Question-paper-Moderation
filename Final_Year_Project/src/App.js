// import './App.css';
// import "../node_modules/bootstrap/dist/css/bootstrap.min.css"; //npm i bootstrap-dark-5 boostrap
// import "../node_modules/bootstrap/dist/js/bootstrap.bundle";
// import "../node_modules/bootstrap/dist/js/bootstrap.bundle.min.js";
// import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
// import SignUp from "./components/SignUp";
// import SignIn from "./components/SignIn";
// import Welcome from "./components/Welcome";
// import Home from "./components/Home";
// function App() {
//   return (

//      <div>
//        <Router>
//       <Routes>
//         <Route  path='/' element={<Home/>} />
//         <Route  path='signup' element={<SignUp/>} />
//         <Route  path='signin' element={<SignIn/>} />
//         <Route  path='welcome' element={<Welcome/>} />
//       </Routes>
//       </Router>
//       </div>

//   );
// }

// export default App;

// App.js
import "./App.css";
import "../node_modules/bootstrap/dist/css/bootstrap.min.css";
import "../node_modules/bootstrap/dist/js/bootstrap.bundle";
import "../node_modules/bootstrap/dist/js/bootstrap.bundle.min.js";
import { BrowserRouter as Router, Route, Routes } from "react-router-dom";
import SignUp from "./components/SignUp";
import SignIn from "./components/SignIn";
import Welcome from "./components/Welcome";
import Home from "./components/Home";
import PdfGenerator from "./components/PdfGenerator"; // Import the new component

function App() {
  return (
    <div>
      <PdfGenerator />
    </div>
  );
}

export default App;
