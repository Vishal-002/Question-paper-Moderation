import React from "react";
// import '../components/assets/css/style.css'
// import '../components/assets/css/bootsnav.css'
import {
  Nav,
  Navbar,
  NavbarBrand,
  NavbarToggler,
  Collapse,
  NavItem,
  Jumbotron,
  Button,
  Modal,
  ModalHeader,
  ModalBody,
  Form,
  FormGroup,
  Input,
  Label,
} from "reactstrap";
function Home() {
  return (
    <div className="App">
      <header>
        <nav className="navbar">
          <div className="logo">Q_Moderation</div>
          <ul className="nav-links">
            <li>
              <a href="signin">Login</a>
            </li>
            <li>
              <a href="signup">SignUp</a>
            </li>
            <li>
              <a href="/contact">Question Paper</a>
            </li>
          </ul>
        </nav>
      </header>
      <div className="content">
        <div className="header-text">
          <h2>
            Welcome<span>,</span> To
            <br /> Question paper Moderation
            <br /> website<span>.</span>{" "}
          </h2>
        </div>
      </div>
    </div>
  );
}
export default Home;

{
  /* <Router>
  <Routes>
    <Route path="/" element={<Home />} />
    <Route path="signup" element={<SignUp />} />
    <Route path="signin" element={<SignIn />} />
    <Route path="welcome" element={<Welcome />} />
    <Route path="pdf-generator" element={<PdfGenerator />} />{" "}
    {/* Add this line */
}
//   </Routes>
// </Router>; */}
