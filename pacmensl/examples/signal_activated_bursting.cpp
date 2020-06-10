//
// Created by Huy Vo on 5/17/19.
//

static char help[] = "Advance_ CME of signal-activated bursting gene expression example.\n\n";

#include<iomanip>
#include <petscmat.h>
#include <petscvec.h>
#include <petscviewer.h>
#include <Sys.h>
#include <armadillo>
#include <cmath>
#include "FspSolverMultiSinks.h"

using arma::dvec;
using arma::Col;
using arma::Row;

using std::cout;
using std::endl;

using namespace pacmensl;
using FspSolver = FspSolverMultiSinks;

void output_marginals(MPI_Comm comm,
                      std::string model_name,
                      std::string part_type,
                      std::string part_approach,
                      std::string constraint_type,
                      DiscreteDistribution &solution);
void output_performance(MPI_Comm comm,
                        std::string model_name,
                        std::string part_type,
                        std::string part_approach,
                        std::string constraint_type,
                        FspSolver &fsp_solver);
void output_time(MPI_Comm comm,
                 std::string model_name,
                 std::string part_type,
                 std::string part_approach,
                 std::string constraint_type,
                 FspSolver &fsp_solver);
void petscvec_to_file(MPI_Comm comm, Vec x, const char *filename);

int main(int argc, char *argv[]) {

  PetscMPIInt ierr, myRank, num_procs;

  ierr = pacmensl::PACMENSLInit(&argc, &argv, help);
  CHKERRQ(ierr);

  // Begin Parallel Fsp context
  {
    MPI_Comm comm;
    MPI_Comm_dup(PETSC_COMM_WORLD, &comm);
    MPI_Comm_size(comm, &num_procs);
    PetscPrintf(comm, "\n ================ \n");

    std::string part_type;
    std::string part_approach;
    // Default problem
    std::string model_name = "Il1beta_expression";
    PetscReal t_final = 4.0;
    PetscReal fsp_tol = 1.0e-4;

    arma::Mat<PetscInt> X0(3, 1, arma::fill::zeros);
    X0(0, 0) = 2;
    arma::Col<PetscReal> p0 = {1.0};

    arma::Row<int> bounds{2, 2, 100};

    arma::Mat<PetscInt> stoich_mat{
        {-1, 1, 0, 0},
        {1, -1, 0, 0},
        {0, 0, 1, -1}
    };

    double k_on{0.1}, k_off{3}, k_rna{79.0}, gamma{0.03};

    double r1{0.29}, r2{0.04}, S_max{16.0};
    auto signal = [&](double t) {
      return S_max * exp(-r1 * t) * (1.0 - exp(-r2 * t));
    };

    double a_gene = 5.0, a_rna = 1.0;
    auto t_fun = [&](double t, int num_coefs, double *outputs, void *args) {
      outputs[0] = a_gene * signal(t) + 1.0;
      outputs[1] = 1.0;
      outputs[2] = a_rna * signal(t) + 1.0;
      outputs[3] = 1.0;
      return 0;
    };

    auto burst_propensity = [&](const int *state, const int k) {
      switch (k) {
        case 0:return k_on * (state[0]);
        case 1:return k_off * (state[1]);
        case 2:return k_rna * (state[1]);
        case 3:return gamma * (state[2]);
        default:return 0.0;
      }
    };

    auto propensity = [&](const int reaction,
                          const int num_species,
                          const int num_states,
                          const int *states,
                          double *values,
                          void *args) {
      int (*X)[3] = ( int (*)[3] ) states;
      for (int i = 0; i < num_states; ++i) {
        values[i] = burst_propensity(X[i], reaction);
      }
      return 0;
    };

    arma::Row<double> expansion_factors{0.0, 0.0, 0.25};

    Model il1b_model;
    il1b_model.stoichiometry_matrix_ = stoich_mat;
    il1b_model.prop_x_ = propensity;
    il1b_model.prop_t_ = t_fun;

    // Default options
    PartitioningType fsp_par_type = PartitioningType::GRAPH;
    PartitioningApproach fsp_repart_approach = PartitioningApproach::REPARTITION;
    ODESolverType fsp_odes_type = CVODE;
    PetscBool output_marginal = PETSC_FALSE;
    PetscBool fsp_log_events = PETSC_FALSE;
    // Read options for fsp
    char opt[100];
    PetscBool opt_set;

    ierr = PetscOptionsGetString(NULL, PETSC_NULL, "-fsp_partitioning_type", opt, 100, &opt_set);
    CHKERRQ(ierr);
    if (opt_set) {
      fsp_par_type = str2part(std::string(opt));
    }

    ierr = PetscOptionsGetString(NULL, PETSC_NULL, "-fsp_repart_approach", opt, 100, &opt_set);
    CHKERRQ(ierr);
    if (opt_set) {
      fsp_repart_approach = str2partapproach(std::string(opt));
    }

    ierr = PetscOptionsGetString(NULL, PETSC_NULL, "-fsp_output_marginal", opt, 100, &opt_set);
    CHKERRQ(ierr);
    if (opt_set) {
      if (strcmp(opt, "1") == 0 || strcmp(opt, "true") == 0) {
        output_marginal = PETSC_TRUE;
      }
    }
    PetscPrintf(comm, "Solving with %d processors.\n", num_procs);

    ierr = PetscOptionsGetString(NULL, PETSC_NULL, "-fsp_log_events", opt, 100, &opt_set);
    CHKERRQ(ierr);
    if (opt_set) {
      if (strcmp(opt, "1") == 0 || strcmp(opt, "true") == 0) {
        fsp_log_events = PETSC_TRUE;
      }
    }

    part_type = part2str(fsp_par_type);
    part_approach = partapproach2str(fsp_repart_approach);
    PetscPrintf(comm, "Partitiniong option %s \n", part2str(fsp_par_type).c_str());
    PetscPrintf(comm, "Repartitoning option %s \n", partapproach2str(fsp_repart_approach).c_str());

    // Solve using adaptive custom constraints
    FspSolverMultiSinks fsp_solver(PETSC_COMM_WORLD, fsp_par_type, fsp_odes_type);
    fsp_solver.SetExpansionFactors(expansion_factors);
    fsp_solver.SetModel(il1b_model);
    fsp_solver.SetInitialDistribution(X0, p0);
    fsp_solver.SetInitialBounds(bounds);
    fsp_solver.SetFromOptions();
    fsp_solver.SetUp();
    DiscreteDistribution solution;

    double t1, t2;
    PetscTime(&t1);
    for (int j{0}; j < 10; ++j) {
      solution = fsp_solver.Solve(t_final, fsp_tol, 0);
      PetscTime(&t2);
      PetscPrintf(comm, "Elapsed time: %.2f \n", t2 - t1);
    }

    if (fsp_log_events) {
      output_time(PETSC_COMM_WORLD, model_name, part_type, part_approach, std::string("adaptive_custom"), fsp_solver);
      output_performance(PETSC_COMM_WORLD,
                         model_name,
                         part_type,
                         part_approach,
                         std::string("adaptive_custom"),
                         fsp_solver);
    }
    if (output_marginal) {
      output_marginals(PETSC_COMM_WORLD,
                       model_name,
                       part_type,
                       part_approach,
                       std::string("adaptive_custom"),
                       solution);
    }

    fsp_solver.ClearState();
  }
  //End Parallel Fsp context
  ierr = pacmensl::PACMENSLFinalize();
  return ierr;
}

void output_marginals(MPI_Comm comm,
                      std::string model_name,
                      std::string part_type,
                      std::string part_approach,
                      std::string constraint_type,
                      DiscreteDistribution &solution) {
  int myRank, num_procs;
  MPI_Comm_rank(comm, &myRank);
  MPI_Comm_size(comm, &num_procs);
  /* Compute the marginal distributions */
  std::vector<arma::Col<PetscReal>> marginals(solution.states_.n_rows);
  for (PetscInt i{0}; i < marginals.size(); ++i) {
    marginals[i] = Compute1DMarginal(solution, i);
  }

  MPI_Comm_rank(PETSC_COMM_WORLD, &myRank);
  if (myRank == 0) {
    for (PetscInt i{0}; i < marginals.size(); ++i) {
      std::string filename =
          model_name + "_marginal_" + std::to_string(i) + "_" +
              std::to_string(num_procs) +
              "_" +
              part_type + "_" + part_approach + "_" + constraint_type + ".dat";
      marginals[i].save(filename, arma::raw_ascii);
    }

//        std::string filename = model_name + "_bounds_" + std::to_string(num_procs) + "_" + part_type + "_" + part_approach + "_" + constraint_type + ".dat";
//        ((StateSetConstrained *) state_set)->GetShapeBounds().save(filename, arma::raw_ascii);
  }
}

void output_time(MPI_Comm comm,
                 std::string model_name,
                 std::string part_type,
                 std::string part_approach,
                 std::string constraint_type,
                 FspSolver &fsp_solver) {
  int myRank, num_procs;
  MPI_Comm_rank(comm, &myRank);
  MPI_Comm_size(comm, &num_procs);

  FspSolverComponentTiming timings = fsp_solver.GetAvgComponentTiming();
  FiniteProblemSolverPerfInfo perf_info = fsp_solver.GetSolverPerfInfo();
  double solver_time = timings.TotalTime;

  if (myRank == 0) {
    {
      std::string filename =
          model_name + "_time_" + std::to_string(num_procs) + "_" + part_type +
              "_" + part_approach + "_" + constraint_type + ".dat";
      std::ofstream file;
      file.open(filename, std::ios_base::app);
      file << solver_time << "\n";
      file.close();
    }
  }
}

void output_performance(MPI_Comm comm,
                        std::string model_name,
                        std::string part_type,
                        std::string part_approach,
                        std::string constraint_type,
                        FspSolver &fsp_solver) {
  int myRank, num_procs;
  MPI_Comm_rank(comm, &myRank);
  MPI_Comm_size(comm, &num_procs);

  FspSolverComponentTiming timings = fsp_solver.GetAvgComponentTiming();
  FiniteProblemSolverPerfInfo perf_info = fsp_solver.GetSolverPerfInfo();
  double solver_time = timings.TotalTime;
  if (myRank == 0) {
    std::string filename =
        model_name + "_time_breakdown_" + std::to_string(num_procs) + "_" + part_type +
            "_" + part_approach + "_" + constraint_type + ".dat";
    std::ofstream file;
    file.open(filename);
    file << "Component, Average processor time (sec), Percentage \n";
    file << "Finite State Subset," << std::scientific << std::setprecision(2)
         << timings.StatePartitioningTime << ","
         << timings.StatePartitioningTime / solver_time * 100.0
         << "\n"
         << "Matrix Generation," << std::scientific << std::setprecision(2)
         << timings.MatrixGenerationTime << ","
         << timings.MatrixGenerationTime / solver_time * 100.0
         << "\n"
         << "Matrix-vector multiplication," << std::scientific << std::setprecision(2)
         << timings.RHSEvalTime << "," << timings.RHSEvalTime / solver_time * 100.0 << "\n"
         << "Others," << std::scientific << std::setprecision(2)
         << solver_time - timings.StatePartitioningTime - timings.MatrixGenerationTime -
             timings.RHSEvalTime << ","
         << (solver_time - timings.StatePartitioningTime - timings.MatrixGenerationTime -
             timings.RHSEvalTime) / solver_time * 100.0
         << "\n"
         << "Total," << solver_time << "," << 100.0 << "\n";
    file.close();

    filename =
        model_name + "_perf_info_" + std::to_string(num_procs) + "_" + part_type +
            "_" + part_approach + "_" + constraint_type + ".dat";
    file.open(filename);
    file << "Model time, ODEs size, Average processor time (sec) \n";
    for (auto i{0}; i < perf_info.n_step; ++i) {
      file << perf_info.model_time[i] << "," << perf_info.n_eqs[i] << ","
           << perf_info.cpu_time[i]
           << "\n";
    }
    file.close();
  }
}

void petscvec_to_file(MPI_Comm comm, Vec x, const char *filename) {
  PetscViewer viewer;
  PetscViewerCreate(comm, &viewer);
  PetscViewerBinaryOpen(comm, filename, FILE_MODE_WRITE, &viewer);
  VecView(x, viewer);
  PetscViewerDestroy(&viewer);
}