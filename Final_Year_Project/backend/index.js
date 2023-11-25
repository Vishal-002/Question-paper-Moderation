const express = require("express");
const mongoose = require("mongoose");
const bodyParser = require("body-parser");
const app = express();
const cors = require("cors");
const bcrypt = require("bcryptjs");
const jwt = require("jsonwebtoken");
const config = require("config");
const port = 5000;
app.use(bodyParser.urlencoded({ extended: false }));
const router = express.Router();
app.use(cors());
// const uri =
//   "mongodb+srv://user2001:test123@cluster0.yrjxwok.mongodb.net/test?retryWrites=true&w=majority";
const uri =
  "mongodb+srv://user2001:test123@cluster0.yrjxwok.mongodb.net/test?retryWrites=true&w=majority";

mongoose.connect(uri);
const connection = mongoose.connection;

connection.once("open", () => {
  console.log("Connected to MongoDB successfully.");
});

const Schema = mongoose.Schema;

const newExercise = new Schema({
  username: { type: String },
  email: { type: String },
  password: { type: String },
});

const Exercise = mongoose.model("document", newExercise);
app.use(express.json());
app.post("/api/register", async (req, res) => {
  const { username, email, password } = req.body;
  // console.log(req.body);
  // Simulate user registration (in a real app, this would interact with a database)
  // For now, just send back a success message
  try {
    const newItem = new Exercise(req.body);
    await newItem.save();
    // res.status(201).json(newItem);
    console.log(req.body);
  } catch (error) {
    console.log(error);
    res.status(500).json({ error: "Unable to save item" });
  }
  res.json({ message: "Registration successful", username, email });
  // res.redirect('/signin');
});
app.post("/api/login", async (req, res) => {
  const { email, password } = req.body;
  console.log(password);

  try {
    const check = await Exercise.findOne({ email: email });
    const user = await Exercise.findOne({ email });
    console.log(user.password);
    if (check) {
      if (user.password == password) {
        res.json({ username: user.username, status: "exist" });
        console.log(user.username);
      } else {
        res.json("pnotexist");
      }
    } else {
      res.json("notexist");
    }
  } catch (e) {
    res.json("fail");
  }
});
app.listen(port, () => {
  console.log(`app listening on port ${port}`);
});
