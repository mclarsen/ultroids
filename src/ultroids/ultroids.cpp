#include <chai/ManagedArray.hpp>
#include <RAJA/RAJA.hpp>
#include <iostream>
#include <math.h>
#include <random>

struct Particle
{
  double m_vx, m_vy, m_vz;
  double m_fx, m_fy, m_fz;
  double m_px, m_py, m_pz;
  double m_mass;
};

void reset(Particle &p)
{
  p.m_fx = 0.f;
  p.m_fy = 0.f;
  p.m_fz = 0.f;
}

void add_force(Particle &a, const Particle &b)
{
  const double dx = a.m_px - b.m_px;
  const double dy = a.m_py - b.m_py;
  const double dz = a.m_pz - b.m_pz;
  double dist_sq = dx * dx + dy * dy + dz * dz;
  if(dist_sq == 0.)
  {
    dist_sq = 1.; // not sure what this value should be
  }

  constexpr double G = 6.673e-11;
  // F = ma
  double f = (G * a.m_mass * b.m_mass) / dist_sq;
  const double inv_dist = 1.0 / sqrt(dist_sq);
  a.m_fx += f * dx * inv_dist;
  a.m_fy += f * dy * inv_dist;
  a.m_fz += f * dz * inv_dist;
}

void advance(Particle &a, const double &delta_t)
{
  const double inv_mass = 1. / a.m_mass;
  a.m_vx += a.m_fx * delta_t * inv_mass;
  a.m_vy += a.m_fy * delta_t * inv_mass;
  a.m_vz += a.m_fz * delta_t * inv_mass;
  a.m_px += a.m_vx * delta_t;
  a.m_py += a.m_vy * delta_t;
  a.m_pz += a.m_vz * delta_t;
}

void print_particle(Particle &p)
{
  std::cout<<"pos: "<<p.m_px<<", "<<p.m_py<<", "<<p.m_pz<<"\n";
  std::cout<<"vel: "<<p.m_vx<<", "<<p.m_vy<<", "<<p.m_vz<<"\n";
  std::cout<<"f: "<<p.m_fx<<", "<<p.m_fy<<", "<<p.m_fz<<"\n";
}
void init_particles(chai::ManagedArray<Particle> &particles)
{
  double lower_bound = 0.;
  double upper_bound = 10.;
  std::uniform_real_distribution<double> unif(lower_bound, upper_bound);
  std::default_random_engine re;
  for(int i = 0; i < particles.size(); ++i)
  {
    particles[i].m_mass = 1.;
    particles[i].m_fx = 0.;
    particles[i].m_fy = 0.;
    particles[i].m_fz = 0.;
    particles[i].m_vx = 0.;
    particles[i].m_vy = 0.;
    particles[i].m_vz = 0.;
    particles[i].m_px = unif(re);
    particles[i].m_py = unif(re);
    particles[i].m_pz = unif(re);
  }
}

#include <iostream>
#include <fstream>
void write_particles(chai::ManagedArray<Particle> &particles)
{
  const int num_particles = particles.size();
  std::ofstream file;
  file.open ("particles.vtk");
  file<<"# vtk DataFile Version 3.0\n";
  file<<"particles\n";
  file<<"ASCII\n";
  file<<"DATASET UNSTRUCTURED_GRID\n";
  file<<"POINTS "<<num_particles<<" double\n";
  for(int i = 0; i < num_particles; ++i)
  {
    file<<particles[i].m_px<<" ";
    file<<particles[i].m_py<<" ";
    file<<particles[i].m_pz<<"\n";
  }

  file<<"CELLS "<<num_particles<<" "<<num_particles * 2<<"\n";
  for(int i = 0; i < num_particles; ++i)
  {
    file<<"1 "<<i<<"\n";
  }

  file<<"CELL_TYPES "<<num_particles<<"\n";
  for(int i = 0; i < num_particles; ++i)
  {
    file<<"1\n";
  }

  file<<"POINT_DATA "<<num_particles<<"\n";
  file<<"SCALAR velocity_mag double"<<num_particles<<"\n";
  file<<"LOOKUP_TABLE default\n";
  for(int i = 0; i < num_particles; ++i)
  {
    const Particle &p = particles[i];
    double mag = sqrt(p.m_vx * p.m_vx + p.m_vy * p.m_vy + p.m_vy + p.m_vz * p.m_vz);;
    file<<mag<<"\n";
  }
  file<<"VECTORS velocity double"<<num_particles<<"\n";
  for(int i = 0; i < num_particles; ++i)
  {
    file<<particles[i].m_vx<<" ";
    file<<particles[i].m_vy<<" ";
    file<<particles[i].m_vz<<"\n";
  }

  file.close();
}


int main(void)
{
  std::cout<<"hello\n";
  const int size = 20;
  const int steps = 10;
  const double delta_t = 0.5;
  chai::ManagedArray<Particle> particles(size);
  init_particles(particles);

  for(int step = 0; step < steps; step++)
  {
    RAJA::forall<RAJA::seq_exec>(RAJA::RangeSegment(0, size), [=] (int i) {
        Particle &particle = particles[i];
        reset(particle);
        for(int p = 0; p < size; p++)
        {
          if(p != i)
          {
            add_force(particle, particles[p]);
          }
        }

        advance(particle, delta_t);
    });
  }
  write_particles(particles);

}
