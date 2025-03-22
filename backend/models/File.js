import mongoose from "mongoose";

const fileSchema = new mongoose.Schema({
  filename: {
    type: String,
    required: true,
  },
  fileType: {
    type: String,
    required: true,
  },
  fileSize: {
    type: Number,
    required: true,
  },
  url: {
    type: String,
    required: true,
  },
  fileId: {
    type: String,
    required: true,
  },
  organization: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'Organization',
    required: true
  },
  status: {
    type: String,
    enum: ['pending', 'uploading', 'success', 'error'],
    default: 'pending'
  },
  uploadDate: {
    type: Date,
    default: Date.now
  }
}, {
  timestamps: true
});

const File = mongoose.model("File", fileSchema);

export default File;