import express from 'express';
import bcrypt from 'bcryptjs';
import jwt from 'jsonwebtoken';
import Organization from '../models/Organization.js';

const router = express.Router();

// Registration endpoint
router.post('/register', async (req, res) => {
  try {
    const { name, email, password, industry, country } = req.body;

    // Check if organization already exists
    const existingOrg = await Organization.findOne({ email });
    if (existingOrg) {
      return res.status(400).json({ message: 'Organization already exists' });
    }

    // Generate unique organizationId
    const organizationId = `ORG${Date.now()}`;

    // Hash password
    const hashedPassword = await bcrypt.hash(password, 10);

    // Create new organization
    const organization = new Organization({
      name,
      email,
      password: hashedPassword,
      organizationId,
      industry,
      country,
      subscription: {
        plan: 'free',
        status: 'active',
        startDate: new Date(),
      },
    });

    await organization.save();

    // Generate JWT token
    const token = jwt.sign(
      { organizationId: organization._id },
      process.env.JWT_SECRET,
      { expiresIn: '24h' }
    );

    // Return success response with token and organization data
    res.status(201).json({
      message: 'Organization registered successfully',
      token,
      organization: {
        id: organization._id,
        name: organization.name,
        email: organization.email,
        industry: organization.industry,
        country: organization.country,
        subscription: organization.subscription,
      },
    });
  } catch (error) {
    console.error('Registration error:', error);
    res.status(500).json({
      message: 'Registration failed',
      error: error.message,
      details: error.errors ? Object.keys(error.errors).map(key => ({
        field: key,
        message: error.errors[key].message
      })) : null
    });
  }
});

// Login endpoint
router.post('/login', async (req, res) => {
  try {
    console.log('Login attempt:', req.body);
    const { email, password } = req.body;

    if (!email || !password) {
      return res.status(400).json({
        message: 'Email and password are required'
      });
    }

    // Find organization
    const organization = await Organization.findOne({ email });
    if (!organization) {
      console.log('Organization not found:', email);
      return res.status(401).json({
        message: 'Invalid email or password'
      });
    }

    // Verify password
    const isValidPassword = await bcrypt.compare(password, organization.password);
    if (!isValidPassword) {
      console.log('Invalid password for:', email);
      return res.status(401).json({
        message: 'Invalid email or password'
      });
    }

    // Update last login
    organization.lastLogin = new Date();
    await organization.save();

    // Generate JWT token
    const token = jwt.sign(
      {
        organizationId: organization._id,
        email: organization.email
      },
      process.env.JWT_SECRET,
      { expiresIn: '24h' }
    );

    // Return success response
    console.log('Login successful for:', email);
    res.json({
      message: 'Login successful',
      token,
      organization: {
        _id: organization._id,
        name: organization.name,
        email: organization.email,
        industry: organization.industry,
        country: organization.country,
        subscription: organization.subscription,
      },
    });
  } catch (error) {
    console.error('Login error:', error);
    res.status(500).json({
      message: 'An error occurred during login',
      error: error.message
    });
  }
});

// Get organization profile
router.get('/profile', async (req, res) => {
  try {
    const token = req.headers.authorization?.split(' ')[1];
    if (!token) {
      return res.status(401).json({ message: 'No token provided' });
    }

    const decoded = jwt.verify(token, process.env.JWT_SECRET);
    const organization = await Organization.findById(decoded.id).select('-password');

    if (!organization) {
      return res.status(404).json({ message: 'Organization not found' });
    }

    res.json(organization);
  } catch (error) {
    console.error('Profile error:', error);
    res.status(500).json({ message: 'Error fetching profile' });
  }
});

export default router; 