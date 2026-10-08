// SPDX-FileCopyrightText: © 2026 PRISMS Center at the University of Michigan
// SPDX-License-Identifier: GNU Lesser General Public Version 2.1

#include "custom_pde.h"

#include <prismspf/core/parse_cmd_options.h>
#include <prismspf/core/problem.h>

using namespace prismspf;

int
main(int argc, char *argv[])
{
  // Initialize MPI
  prismspf::MPIInitFinalize mpi_init(argc, argv);

  // Parse the command line options (if there are any) to get the name of the input
  // file
  ParseCMDOptions cli_options(argc, argv);

  constexpr unsigned int dim        = 2;
  constexpr unsigned int degree     = 2;

  // For this example, we allow the number of order parameters to vary
  // based on user inputs. There are many ways to do this. One way is to create a
  // "FieldRegistry" class that manages the list of fields, solve blocks, and their
  // indices. For this example, we will make the CustomPDE object "own" the list of
  // field attributes and solve blocks. We begin by constructing our CustomPDE.
  // See the comments in the CustomPDE constructor (customPDE.h)
  UserInputParameters<dim>       user_inputs(cli_options.get_parameters_filename());
  PhaseFieldTools<dim>           pf_tools;
  CustomPDE<dim, degree, double> pde_operator(user_inputs, pf_tools);

  // The CustomPDE constructor creates the list of field attributes and solve blocks.
  // We created "getter" functions in CustomPDE to retrieve these, so that we can
  // pass them to the PRISMS-PF Problem constructor.
  pde_operator.print_field_info();

  Problem<dim, degree, double> problem(pde_operator.get_field_attributes(),
                                       pde_operator.get_solve_blocks(),
                                       user_inputs,
                                       pf_tools,
                                       pde_operator);
  problem.solve();

  return 0;
}
