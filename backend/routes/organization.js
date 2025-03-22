const express = require("express")
const router = express.Router()
const multer = require("multer")
const cloudinary = require("cloudinary").v2
const { CloudinaryStorage } = require("multer-storage-cloudinary")
const Organization = require("../models/Organization")
const auth = require("../middleware/auth")

// Configure Cloudinary
cloudinary.config({
  cloud_name: process.env.CLOUDINARY_CLOUD_NAME,
  api_key: process.env.CLOUDINARY_API_KEY,
  api_secret: process.env.CLOUDINARY_API_SECRET,
})

// Configure multer for file upload
const storage = new CloudinaryStorage({
  cloudinary: cloudinary,
  params: {
    folder: "organization-profiles",
    allowed_formats: ["jpg", "jpeg", "png", "gif"],
    transformation: [{ width: 500, height: 500, crop: "fill" }],
  },
})

const upload = multer({ storage: storage })

// Upload organization profile photo
router.post("/upload-photo", auth, upload.single("profilePhoto"), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ message: "No file uploaded" })
    }

    // Update organization with new profile image URL
    const organization = await Organization.findByIdAndUpdate(
      req.user.organizationId,
      { profileImageUrl: req.file.path },
      { new: true }
    )

    res.json({
      message: "Profile photo uploaded successfully",
      profileImageUrl: req.file.path,
      organization,
    })
  } catch (error) {
    console.error("Error uploading profile photo:", error)
    res.status(500).json({ message: "Error uploading profile photo" })
  }
})

// Get organization details
router.get("/", auth, async (req, res) => {
  try {
    const organization = await Organization.findById(req.user.organizationId)
    if (!organization) {
      return res.status(404).json({ message: "Organization not found" })
    }
    res.json(organization)
  } catch (error) {
    console.error("Error fetching organization:", error)
    res.status(500).json({ message: "Error fetching organization details" })
  }
})

module.exports = router 