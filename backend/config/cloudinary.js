import { v2 as cloudinary } from "cloudinary";
import dotenv from "dotenv";

dotenv.config();

// Configure Cloudinary
cloudinary.config({
  cloud_name: process.env.CLOUDINARY_CLOUD_NAME,
  api_key: process.env.CLOUDINARY_API_KEY,
  api_secret: process.env.CLOUDINARY_API_SECRET,
});

// Test the configuration
cloudinary.api.ping()
  .then(result => {
    console.log('Cloudinary connection successful');
  })
  .catch(error => {
    console.error('Cloudinary connection failed:', error);
  });

export default cloudinary;