const express = require("express");
const cors = require("cors");
require("dotenv").config();

const app = express();

const PORT = process.env.PORT || 5000;

app.use(cors());
app.use(express.json());

app.get("/api/health", (req, res) => {
  res.json({
    success: true,
    service: "SeeBeyond Backend",
    status: "running"
  });
});

app.listen(PORT, () => {
  console.log(`SeeBeyond backend running on http://localhost:${PORT}`);
});

