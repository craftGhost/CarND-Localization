/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;
using std::normal_distribution;
std::default_random_engine gen;

double multiv_prob(double sig_x, double sig_y, double x_obs, double y_obs,
                   double mu_x, double mu_y);

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  
  num_particles = 50;  // TODO: Set the number of particles
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);
  
  for (int i=0; i<num_particles; ++i) {
    struct Particle p;
    p.id = i;
    p.x = dist_x(gen);
    p.y = dist_y(gen);  
    p.theta = dist_theta(gen);
    
    p.weight = 1.0;
    weights.push_back(p.weight);
    particles.push_back(p);
  }
  
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  normal_distribution<double> d_x(0, std_pos[0]);
  normal_distribution<double> d_y(0, std_pos[1]);
  normal_distribution<double> d_theta(0, std_pos[2]);
  
  for(unsigned int i=0; i<particles.size(); ++i) {
    if (fabs(yaw_rate) < 0.0001) {
      particles[i].x += velocity*cos(particles[i].theta)*delta_t;
      particles[i].y += velocity*sin(particles[i].theta)*delta_t;
    }
    else {
      particles[i].x += velocity*(sin(particles[i].theta+yaw_rate*delta_t)-sin(particles[i].theta))/yaw_rate;
      particles[i].y += velocity*(cos(particles[i].theta)-cos(particles[i].theta+yaw_rate*delta_t))/yaw_rate;
      particles[i].theta += yaw_rate*delta_t;
    }
    
    particles[i].x += d_x(gen);
    particles[i].y += d_y(gen);
    particles[i].theta += d_theta(gen);
  }

}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
  for (unsigned int i=0; i<observations.size(); ++i) {
    double min_distance = 10000.0;
    int min_id = -1;
    double distance;
    for (unsigned int j=0; j<predicted.size(); ++j) {
      distance = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);
      if (distance<min_distance) {
        min_distance = distance;
        min_id = predicted[j].id;
      }
    }
    observations[i].id = min_id;
  }
  
}

double multiv_prob(double sig_x, double sig_y, double x_obs, double y_obs,
                   double mu_x, double mu_y) {
  
  double gauss_norm;
  gauss_norm = 1 / (2 * M_PI * sig_x * sig_y);

  double exponent;
  exponent = (pow(x_obs - mu_x, 2) / (2 * pow(sig_x, 2)))
               + (pow(y_obs - mu_y, 2) / (2 * pow(sig_y, 2)));
  
  double weight;
  weight = gauss_norm * exp(-exponent);
  return weight;
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
  double weight_sum = 0.0;
  for (unsigned int i=0; i<particles.size(); ++i) {
    double p_x = particles[i].x;
    double p_y = particles[i].y;
    double p_theta = particles[i].theta;
    particles[i].weight = 1.0;
    vector<LandmarkObs> predict;
    
    for (unsigned int j=0; j<map_landmarks.landmark_list.size(); ++j) {
      double distance = dist(p_x, p_y, map_landmarks.landmark_list[j].x_f, map_landmarks.landmark_list[j].y_f);
      if (distance <= sensor_range) {
        LandmarkObs ob;
        ob.x = map_landmarks.landmark_list[j].x_f;
        ob.y = map_landmarks.landmark_list[j].y_f;
        ob.id = map_landmarks.landmark_list[j].id_i;
        predict.push_back(ob);
      }
    }
    
    vector<LandmarkObs> transformed_obs;
    for (unsigned int k=0; k<observations.size(); ++k) {
      LandmarkObs trans_ob;
      trans_ob.x = p_x + cos(p_theta)*observations[k].x - sin(p_theta)*observations[k].y;
      trans_ob.y = p_y + sin(p_theta)*observations[k].x + cos(p_theta)*observations[k].y;
      trans_ob.id = observations[k].id;
      transformed_obs.push_back(trans_ob);
    }
    
    dataAssociation(predict, transformed_obs);
    
    for (unsigned int m=0; m<transformed_obs.size(); ++m) {
      for (unsigned int n=0; n<predict.size(); ++n) {
        if (transformed_obs[m].id == predict[n].id) {
          double t_weight = multiv_prob(std_landmark[0], std_landmark[1], predict[n].x, predict[n].y, transformed_obs[m].x, transformed_obs[m].y);
          particles[i].weight *= t_weight;
        }
      }
    }
    weight_sum += particles[i].weight;
  }
  
  for (unsigned int i=0; i<particles.size(); ++i) {
    particles[i].weight /= weight_sum;
    weights[i] = particles[i].weight;
  }
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  int N = particles.size();
  vector<Particle> new_particles;
  
  std::discrete_distribution<int> d(weights.begin(), weights.end());
  for (int i=0; i<N; ++i) {
    int idx = d(gen);
    new_particles.push_back(particles[idx]);
  }
  particles = new_particles;
  
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}