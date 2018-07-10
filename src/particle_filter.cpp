/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

  // Set number of particles
  num_particles = 100;

  // Create random number generator
  default_random_engine gen;

  // normal (Gaussian) distribution, give standard deviation
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);


  for (int i = 0; i < num_particles; i++) {
    Particle particle;
    particle.id = i;
    particle.x = dist_x(gen);
    particle.y = dist_y(gen);
    particle.theta = dist_theta(gen);
    particle.weight = 1.0;
    
    weights.push_back(particle.weight);
    particles.push_back(particle);
    
  }

  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	
	// Create random number generator
  default_random_engine gen;
	
	for (int i = 0; i < particles.size(); i++) {
	  double pred_x;
	  double pred_y;
	  double pred_theta;
	
	  // prevent divide by zero
	  if (fabs(yaw_rate) < 1000000000) {
	    pred_x = particles[i].x + velocity * delta_t * cos(particles[i].theta);
	    pred_y = particles[i].y + velocity * delta_t * sin(particles[i].theta);
	    pred_theta = particles[i].theta;
	  }
	  else {
	    pred_x = particles[i].x + velocity/yaw_rate * (sin(particles[i].theta + yaw_rate*delta_t) - sin(particles[i].theta));
      pred_y = particles[i].y + velocity/yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate*delta_t));
      pred_theta = particles[i].theta + (yaw_rate * delta_t);
	  }
	  
	  // normal (Gaussian) distribution, give standard deviation
    normal_distribution<double> dist_x(pred_x, std_pos[0]);
    normal_distribution<double> dist_y(pred_y, std_pos[1]);
    normal_distribution<double> dist_theta(pred_theta, std_pos[2]);
	  
	  // upadate particle
	  particles[i].x = dist_x(gen);
	  particles[i].y = dist_y(gen);
	  particles[i].theta = dist_theta(gen);
	}
}

void ParticleFilter::dataAssociation(double sensor_range,
                                     std::vector<LandmarkObs> observations,
                                     const Map &map_landmarks) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

  // transform observations
  for (int i = 0; i < particles.size(); i++) {
    particles[i].observations.clear();
  
    for (int j = 0; j < observations.size(); j++) {
      particles[i].observations.push_back(array<double, 2> {
        observations[j].x * cos(particles[i].theta) - observations[j].y * sin(particles[i].theta) + particles[i].x,
        observations[j].x * sin(particles[i].theta) + observations[j].y * cos(particles[i].theta) + particles[i].y});
    }
  }
  
  // associate observations with landmarks
  for (int i = 0; i < particles.size(); i++) {
    particles[i].landmarks.clear();
    
    for (int j = 0; j < particles[i].observations.size(); j++) {
      Map::single_landmark_s closest_neighbor;
      double min = numeric_limits<double>::max();
      
      for (int k = 0; k < map_landmarks.landmark_list.size(); k++) {
      
        if (fabs(map_landmarks.landmark_list[k].x_f - particles[i].observations[j][0]) <= sensor_range && 
            fabs(map_landmarks.landmark_list[k].y_f - particles[i].observations[j][1]) <= sensor_range) {
            
          double distance = dist(particles[i].observations[j][0],
                                 particles[i].observations[j][1],
                                 map_landmarks.landmark_list[k].x_f,
                                 map_landmarks.landmark_list[k].y_f);

          if (distance < min) {
            min = distance;
            closest_neighbor = map_landmarks.landmark_list[k];
          }
         }
      }
      particles[i].landmarks.push_back(closest_neighbor);
      particles[i].associations.push_back(closest_neighbor.id_i);
    }
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
  
  double weight;
  double normalize = 0;
  dataAssociation(sensor_range, observations, map_landmarks);
  weights.clear();
  
  for (int i = 0; i < particles.size(); i++) {
    weight = 1;
    
    for (int j = 0; j < particles[i].landmarks.size(); j++) {
      // gaussian normalization inputs
      double sig_x= std_landmark[0];
      double sig_y= std_landmark[1];
      double x_obs = particles[i].observations[j][0];
      double y_obs = particles[i].observations[j][1];
      double mu_x = particles[i].landmarks[j].x_f;
      double mu_y = particles[i].landmarks[j].y_f;
      
      // calculate gaussian normalization
      double gauss_norm= (1/(2 * M_PI * sig_x * sig_y));
      double exponent= ((x_obs - mu_x)*(x_obs - mu_x))/(2 * sig_x*sig_x) + 
                       ((y_obs - mu_y)*(y_obs - mu_y))/(2 * sig_y*sig_y);
      weight = weight + gauss_norm * exp(-exponent);
    }
    
    for (int j = 0; j < particles.size(); j++) {
      particles[j].weight = particles[j].weight/normalize;
      weights.push_back(particles[j].weight);
    }
  }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

  // Create random number generator
  default_random_engine gen;

  // Create discrete distribution
  discrete_distribution<int> dist(weights.begin(), weights.end());
  
  // resampled particles
	vector<Particle> resampled_particles;
	for (int i = 0; i < num_particles; i++) {
	  resampled_particles.push_back(particles[dist(gen)]);
	}
	
	particles.clear();
	particles = resampled_particles;
	num_particles = particles.size();
	
	// Particle ID update
	for (int i = 0; i < num_particles; i++) {
	  particles[i].id = i;
	}
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
