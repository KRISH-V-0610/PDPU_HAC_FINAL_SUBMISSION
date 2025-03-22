import express from "express";
import multer from "multer";
import cloudinary from "../config/cloudinary.js";
import File from "../models/File.js";
import Organization from "../models/Organization.js";
import fs from "fs"; // Import the fs module
import auth from "../middleware/auth.js";

const router = express.Router();

// Configure multer for memory storage
const upload = multer({
  storage: multer.memoryStorage(),
  limits: {
    fileSize: 10 * 1024 * 1024, // 10MB limit
  },
  fileFilter: (req, file, cb) => {
    // Accept only specific file types
    const allowedTypes = [
      'application/pdf',
      'application/msword',
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
      'application/vnd.ms-excel',
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
      'image/jpeg',
      'image/png',
      'image/jpg'
    ];
    if (allowedTypes.includes(file.mimetype)) {
      cb(null, true);
    } else {
      cb(new Error('Invalid file type. Only PDF, DOC, DOCX, XLS, XLSX, JPEG, PNG, and JPG files are allowed.'));
    }
  }
});

// Generate a unique file ID
const generateFileId = () => {
  return 'FILE-' + Math.random().toString(36).substr(2, 9).toUpperCase();
};

// Format filename
const formatFilename = (originalname) => {
  // Remove file extension
  const nameWithoutExt = originalname.replace(/\.[^/.]+$/, "");
  // Get file extension
  const ext = originalname.split('.').pop();
  // Format the name: capitalize first letter of each word, replace spaces with underscores
  const formattedName = nameWithoutExt
    .split(/[\s-_]+/)
    .map(word => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase())
    .join('_');
  // Add timestamp to make it unique
  const timestamp = new Date().toISOString().split('T')[0];
  return `${formattedName}_${timestamp}.${ext}`;
};

// Upload a file
router.post("/upload", auth, upload.single("file"), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ message: "No file uploaded" });
    }

    if (!req.user || !req.user.organizationId) {
      return res.status(401).json({ message: "Authentication required" });
    }

    // Create a temporary file path
    const tempFilePath = `temp_${Date.now()}_${req.file.originalname}`;
    
    // Write the buffer to a temporary file
    fs.writeFileSync(tempFilePath, req.file.buffer);

    // Upload to Cloudinary
    const uploadResponse = await cloudinary.uploader.upload(tempFilePath, {
      resource_type: "auto",
      folder: "compliance-files",
    });

    // Delete the temporary file
    fs.unlinkSync(tempFilePath);

    // Create file document with required fields
    const file = new File({
      filename: req.file.originalname,
      fileType: req.file.mimetype,
      fileSize: req.file.size,
      url: uploadResponse.secure_url,
      fileId: uploadResponse.public_id,
      status: "success",
      organization: req.user.organizationId // Add organization reference
    });

    await file.save();

    res.status(201).json({
      message: "File uploaded successfully",
      file: {
        _id: file._id,
        filename: file.filename,
        fileType: file.fileType,
        fileSize: file.fileSize,
        url: file.url,
        status: file.status
      }
    });
  } catch (error) {
    console.error("Error uploading file:", error);
    // Clean up temporary file if it exists
    try {
      if (req.file) {
        const tempFilePath = `temp_${Date.now()}_${req.file.originalname}`;
        if (fs.existsSync(tempFilePath)) {
          fs.unlinkSync(tempFilePath);
        }
      }
    } catch (cleanupError) {
      console.error("Error cleaning up temporary file:", cleanupError);
    }
    res.status(500).json({ 
      message: "Error uploading file", 
      error: error.message 
    });
  }
});

// Get all files
router.get("/", async (req, res) => {
  try {
    const files = await File.find().sort({ uploadDate: -1 });
    res.status(200).json(files);
  } catch (err) {
    console.error(err);
    res.status(500).json({ message: "Error fetching files" });
  }
});

// Get files for a specific organization
router.get("/organization/:organizationId", async (req, res) => {
  try {
    // Validate MongoDB ObjectID format
    const { organizationId } = req.params;
    if (!organizationId || !organizationId.match(/^[0-9a-fA-F]{24}$/)) {
      return res.status(400).json({
        message: "Invalid organization ID format",
        error: "Organization ID must be a valid MongoDB ObjectId"
      });
    }

    // Verify organization exists
    const organization = await Organization.findById(organizationId);
    if (!organization) {
      return res.status(404).json({ message: "Organization not found" });
    }

    const files = await File.find({ organization: organizationId })
      .sort({ uploadDate: -1 });

    res.status(200).json(files);
  } catch (error) {
    console.error("Error fetching organization files:", error);
    res.status(500).json({ 
      message: "Error fetching organization files",
      error: error.message 
    });
  }
});

// Get organization by ID
router.get("/organizations/:organizationId", async (req, res) => {
  try {
    // Validate MongoDB ObjectID format
    const { organizationId } = req.params;
    if (!organizationId || !organizationId.match(/^[0-9a-fA-F]{24}$/)) {
      return res.status(400).json({
        message: "Invalid organization ID format",
        error: "Organization ID must be a valid MongoDB ObjectId"
      });
    }

    const organization = await Organization.findById(organizationId);
    if (!organization) {
      return res.status(404).json({ message: "Organization not found" });
    }

    res.status(200).json(organization);
  } catch (error) {
    console.error("Error fetching organization:", error);
    res.status(500).json({
      message: "Error fetching organization",
      error: error.message
    });
  }
});

// Delete a file
router.delete("/:fileId", async (req, res) => {
  try {
    // Check if fileId is a valid MongoDB ObjectId
    if (!req.params.fileId.match(/^[0-9a-fA-F]{24}$/)) {
      return res.status(400).json({
        message: "Invalid file ID format",
        error: "File ID must be a valid MongoDB ObjectId"
      });
    }

    const file = await File.findById(req.params.fileId);
    if (!file) {
      return res.status(404).json({ message: "File not found" });
    }

    // Delete from Cloudinary
    await cloudinary.uploader.destroy(file.fileId);

    // Delete from MongoDB
    await File.findByIdAndDelete(req.params.fileId);

    res.status(200).json({ message: "File deleted successfully" });
  } catch (err) {
    console.error("Delete error:", err);
    res.status(500).json({
      message: "Error deleting file",
      error: err.message
    });
  }
});

// Check and create organization if needed
router.post("/check-organization", async (req, res) => {
  try {
    const { organizationId } = req.body;

    if (!organizationId) {
      return res.status(400).json({
        message: "Organization ID is required",
        error: "Missing organizationId in request body"
      });
    }

    // Check if organization exists
    let organization = await Organization.findOne({ 
      $or: [
        { organizationId: organizationId },
        { _id: organizationId }
      ]
    });

    if (!organization) {
      console.log('Creating new organization with ID:', organizationId);
      
      // Create new organization if it doesn't exist
      organization = new Organization({
        name: "Test Organization",
        email: "test@example.com",
        password: "test123", // This should be properly hashed in production
        organizationId: organizationId,
        industry: "technology", // Using lowercase to match enum
        country: "India",
        isActive: true,
        subscription: {
          plan: "free",
          status: "active",
          startDate: new Date()
        }
      });

      try {
        await organization.save();
        console.log('Created new organization:', organization);
      } catch (saveError) {
        console.error('Error saving organization:', saveError);
        return res.status(400).json({
          message: 'Error creating organization',
          error: saveError.message,
          details: saveError.errors
        });
      }
    } else {
      console.log('Found existing organization:', organization._id);
    }

    res.json({
      exists: true,
      organization
    });
  } catch (error) {
    console.error('Error checking/creating organization:', error);
    res.status(500).json({
      message: 'Error checking organization',
      error: error.message,
      details: error.errors
    });
  }
});

export default router;




// import express from "express";
// import multer from "multer";
// import { v2 as cloudinary } from "cloudinary";
// import File from "../models/File.js";
// import fs from "fs";

// const router = express.Router();
// const upload = multer({ dest: "uploads/" });

// // Upload file to Cloudinary and make it public
// router.post("/upload", upload.single("file"), async (req, res) => {
//   try {
//     if (!req.file) {
//       return res.status(400).json({ message: "No file uploaded" });
//     }

//     // Upload the file to Cloudinary
//     const result = await cloudinary.uploader.upload(req.file.path, {
//       resource_type: "auto", // Automatically detect file type
//       public_id: req.file.originalname, // Use the file name as the public ID
//       access_mode: "public", // Make the file publicly accessible
//     });

//     // Save file metadata to MongoDB
//     const newFile = new File({
//       filename: req.file.originalname,
//       url: result.secure_url,
//     });
//     await newFile.save();

//     // Delete the temporary file
//     fs.unlinkSync(req.file.path);

//     res.status(201).json({
//       message: "File uploaded successfully",
//       file: newFile,
//     });
//   } catch (err) {
//     console.error("Upload error:", err);
//     res.status(500).json({ message: "Error uploading file" });
//   }
// });

// export default router;