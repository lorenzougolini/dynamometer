import { useState } from 'react';
import axios from "axios";
import { useDropzone } from "react-dropzone" ;
import Graph  from './Graph';
import API_BASE_URL from './config';
import './App.css'

function App() {
  const [video, setVideo] = useState(null);
  const [data, setData] = useState([]);
  const [isLoading, setIsLoading] = useState(false);

  const { getRootProps, getInputProps } = useDropzone({
    accept: {"video/*": []},
    onDrop: (acceptedFiles) => setVideo(acceptedFiles[0]),
  });

  const uploadVideo = async () => {
    if (!video) return alert("Please select a video first.");
    
    const formData = new FormData();
    formData.append("video", video);

    setIsLoading(true);

    try {
      const response = await axios.post(`${API_BASE_URL}/upload`, formData);
      const formattedData = Object.entries(response.data.data).map(([time, force]) => ({
        time: parseFloat(time),
        force: parseFloat(force),
      }));
      setData(formattedData);
    } catch (error) {
      console.error("Upload failed:", error);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="p-8 flex flex-col items-center">
      <h1 className="text-xl font-bold mb-4">Dynamometer Analysis</h1>

      <div {...getRootProps()} className="border-2 border-dashed p-10 cursor-pointer text-center mb-4">
        <input {...getInputProps()} />
        {video ? <p>{video.name}</p> : <p>Drag & drop a video or click to upload</p>}
      </div>

      <div className="inline-flex items-center mb-6">
        <button onClick={uploadVideo} className="bg-blue-500 text-white px-4 py-2 rounded">
          Upload & Process
        </button>
        {isLoading && <div className="loader ml-4"></div>}
      </div>

      <Graph data={data} />
    </div>
  )
}

export default App
