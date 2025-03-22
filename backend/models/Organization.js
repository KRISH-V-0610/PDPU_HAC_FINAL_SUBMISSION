import mongoose from 'mongoose';

const organizationSchema = new mongoose.Schema({
  name: {
    type: String,
    required: true,
    trim: true
  },
  email: {
    type: String,
    required: true,
    unique: true,
    trim: true,
    lowercase: true
  },
  password: {
    type: String,
    required: true
  },
  industry: {
    type: String,
    required: true,
    enum: ["technology", "finance", "healthcare", "retail", "other"],
  },
  country: {
    type: String,
    required: true
  },
  createdAt: {
    type: Date,
    default: Date.now
  }
}, {
  timestamps: true
});

const Organization = mongoose.model('Organization', organizationSchema);

export default Organization; 