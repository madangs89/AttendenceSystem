import express from "express";
import dotenv from "dotenv";
import cors from "cors";
dotenv.config();
const app = express();

app.use(express.json());
app.use(express.urlencoded({ extended: true }));

app.use(
  cors({
    origin: "*",
  }),
);

app.get("/", (req, res) => {
  res.send("<h1>Hello, World!</h1>");
});

app.listen(3000, () => {
  console.log("Server is running on port 3000");
});
