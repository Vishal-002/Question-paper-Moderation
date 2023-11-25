import React, { useState } from "react";
import axios from "axios";
import { saveAs } from "file-saver";

const PdfGenerator = () => {
  const [file1, setFile1] = useState(null);
  const [file2, setFile2] = useState(null);

  const handleFile1Change = (e) => {
    setFile1(e.target.files[0]);
  };

  const handleFile2Change = (e) => {
    setFile2(e.target.files[0]);
  };

  const generatePdf = async () => {
    try {
      const formData = new FormData();
      formData.append("file1", file1);
      formData.append("file2", file2);

      // Make an API call to your server
      const response = await axios.post(
        "http://127.0.0.1:5002/process_pdfs",
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data",
          },
          responseType: "blob",
        }
      );

      // Use file-saver to save the file
      saveAs(response.data, "output.pdf");
    } catch (error) {
      console.error("Error generating PDF:", error);
    }
  };

  return (
    <div>
      <h2>Pdf Generator</h2>
      <div>
        <label htmlFor="file1">Select PDF File 1:</label>
        <input
          type="file"
          id="file1"
          accept=".pdf"
          onChange={handleFile1Change}
        />
      </div>
      <div>
        <label htmlFor="file2">Select PDF File 2:</label>
        <input
          type="file"
          id="file2"
          accept=".pdf"
          onChange={handleFile2Change}
        />
      </div>
      <button onClick={generatePdf}>Generate PDF</button>
    </div>
  );
};

export default PdfGenerator;
