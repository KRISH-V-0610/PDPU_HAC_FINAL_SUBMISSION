import mongoose from 'mongoose';
import Organization from '../models/Organization.js';

// Connect to MongoDB
mongoose.connect(process.env.MONGO_URI || 'mongodb://localhost:27017/fintech')
  .then(() => console.log('MongoDB connected'))
  .catch(err => {
    console.error('MongoDB connection error:', err);
    process.exit(1);
  });

const createTestOrganization = async () => {
  try {
    // Test organization data
    const testOrg = {
      _id: "655df6f45c7253dc6e63a6ab", // This matches the hardcoded ID in frontend
      name: "Test Organization",
      email: "test@example.com",
      password: "test123", // This should be properly hashed in production
      organizationId: "ORG-TEST-123",
      industry: "technology",
      country: "India",
      isActive: true,
      subscription: {
        plan: "free",
        status: "active",
        startDate: new Date()
      }
    };

    // Check if organization exists
    const existingOrg = await Organization.findById(testOrg._id);
    
    if (existingOrg) {
      console.log('Test organization already exists:', existingOrg._id);
      process.exit(0);
    }

    // Create new organization
    const organization = new Organization(testOrg);
    await organization.save();
    
    console.log('Created test organization:', organization._id);
    process.exit(0);
  } catch (error) {
    console.error('Error creating test organization:', error);
    process.exit(1);
  }
};

createTestOrganization(); 