// SPDX-FileCopyrightText: © 2026 PRISMS Center at the University of Michigan
// SPDX-License-Identifier: GNU Lesser General Public Version 2.1

#include <prismspf/core/pde_operator_base.h>

PRISMS_PF_BEGIN_NAMESPACE

template <unsigned int dim, unsigned int degree, typename number>
class CustomPDE : public PDEOperatorBase<dim, degree, number>
{
public:
  using ScalarValue = dealii::VectorizedArray<number>;
  using ScalarGrad  = dealii::Tensor<1, dim, ScalarValue>;
  using ScalarHess  = dealii::Tensor<2, dim, ScalarValue>;
  using VectorValue = dealii::Tensor<1, dim, ScalarValue>;
  using VectorGrad  = dealii::Tensor<2, dim, ScalarValue>;
  using VectorHess  = dealii::Tensor<3, dim, ScalarValue>;
  using PDEOperatorBase<dim, degree, number>::get_user_inputs;
  using PDEOperatorBase<dim, degree, number>::get_pf_tools;

  CustomPDE(const UserInputParameters<dim> &_user_inputs, PhaseFieldTools<dim> &_pf_tools)
    : PDEOperatorBase<dim, degree, number>(_user_inputs, _pf_tools)
    , m_well(get_user_inputs().user_constants.get_double("m_well"))
    , kappa(get_user_inputs().user_constants.get_double("kappa"))
    , alpha(get_user_inputs().user_constants.get_double("alpha"))
    , kinetic_coef(get_user_inputs().user_constants.get_double("kinetic_coef"))
    , num_grains(get_user_inputs().user_constants.get_int("num_order_parameters"))
  {
    // In the constructor for this application, we create the field attributes list
    // and the list of solve blocks. This allows us to have a variable number of
    // fields depending on the user inputs (this is not the only way to do this).

    // Declare the order parameter fields
    unsigned int current_solve_block_id = 0;
    for (unsigned int i = 0; i < num_grains; i++)
      {
        std::string     field_name = "n" + std::to_string(i);
        FieldAttributes attr(field_name, Scalar);

        // To mark grains for remapping, give them a grain_reassignment_block_id
        // Fields with the same grain_reassignment_block_id will be grouped together
        // and treated as a set of order parameters for grain reassignment.
        // Then in the input parameters file, set use grain remapping = true
        attr.grain_reassignment_block_id = 1;
        field_attributes.push_back(attr);

        // Due to a deal.II bug, we can't have more than 20 fields per solve block.
        // Work around this by dividing the order parameters into multiple solve blocks.
        if (i % max_fields_per_solve_block == 0)
          {
            current_solve_block_id++;
          }
        field_to_block_id[i] = current_solve_block_id;
      }
    
    num_explicit_blocks = current_solve_block_id;
    postprocess_block_id = num_explicit_blocks + 1;

    // Declare the postprocessing fields
    field_attributes.emplace_back("sum2op", Scalar);
    field_attributes.emplace_back("F", Scalar);
    field_attributes.emplace_back("max_op_id", Scalar);
    field_to_block_id[num_grains] = postprocess_block_id;
    field_to_block_id[num_grains + 1] = postprocess_block_id;
    field_to_block_id[num_grains + 2] = postprocess_block_id;

    // Create the solve blocks
    // The order parameters depend on the old value and gradient of the order parameters
    const Dependency old_1_val_and_grad(EvalFlags::nothing,
                                        EvalFlags::nothing,
                                        {EvalFlags::values | EvalFlags::gradients});

    for (int block = 1; block <= num_explicit_blocks; block++)
      {
        SolveBlock exp_block;
        exp_block.id = block;
        for (unsigned int i = 0; i < num_grains; i++)
          {
            exp_block.dependencies_rhs[i] = old_1_val_and_grad;
            if (block == field_to_block_id[i])
              {
                exp_block.field_indices.insert(i);
              }
          }
        solve_blocks.push_back(exp_block);
      }

    SolveBlock pp_block;
    pp_block.id            = postprocess_block_id;
    pp_block.solve_type    = Explicit;
    pp_block.solve_timing  = PostProcess;
    pp_block.field_indices = {num_grains, num_grains + 1, num_grains + 2};

    // The postprocessing block depends on the current values and gradients of the order
    // parameters
    const Dependency current_val_and_grad(EvalFlags::values | EvalFlags::gradients);
    for (unsigned int i = 0; i < num_grains; i++)
      {
        pp_block.dependencies_rhs[i] = current_val_and_grad;
      }
    
    solve_blocks.push_back(pp_block);
  };

  void
  print_field_info()
  {
    Logger::instance() << "\n-----------------------------------------"
                       << std::endl;
    Logger::instance() << " Field Information - Grain Growth Model\n";

    for (unsigned int i = 0; i < field_attributes.size(); i++) {
      Logger::instance() << "  " << i << ": " << field_attributes[i].name
                         << ",   solve_block=" << field_to_block_id[i]
                         << std::endl;
    }
    Logger::instance() << "-----------------------------------------"
                       << std::endl;
  }

  [[nodiscard]] unsigned int
  get_block_id_for_field(unsigned int field_index) const
  {
    return field_to_block_id[field_index];
  }

  [[nodiscard]] bool
  is_explicit_block(unsigned int block_id) const
  {
    return block_id <= num_explicit_blocks;
  }

  [[nodiscard]] bool
  is_postprocessing_block(unsigned int block_id) const
  {
    return block_id == postprocess_block_id;
  }

  [[nodiscard]] unsigned int
  solve_block_start(unsigned int block_id) const
  {
    ASSERT(block_id > 0);
    return (block_id - 1) * max_fields_per_solve_block;
  }

  [[nodiscard]] unsigned int
  solve_block_end(unsigned int block_id) const
  {
    return std::min(block_id * max_fields_per_solve_block, num_grains);
  }

  std::vector<FieldAttributes> &
  get_field_attributes()
  {
    return field_attributes;
  }

  std::vector<SolveBlock> &
  get_solve_blocks()
  {
    return solve_blocks;
  }

private:
  void
  set_initial_condition([[maybe_unused]] const unsigned int       &index,
                        [[maybe_unused]] const unsigned int       &component,
                        [[maybe_unused]] const dealii::Point<dim> &point,
                        [[maybe_unused]] number                   &scalar_value,
                        [[maybe_unused]] number &vector_component_value) const override
  {
    // For this example, the initial grain structure is given by a set of
    // overlapping circles defined by a hyperbolic tangent function. The
    // center of each circle is given by "center_list" and the radius by
    // "radius"

    if (index < 5)
      {
        const dealii::Tensor<1, dim> &mesh_size =
          get_user_inputs().spatial_discretization.rectangular_mesh.size;

        std::vector<dealii::Point<dim>> center_list;

        std::vector<number> radius = {0.14,
                                      0.14,
                                      0.14,
                                      0.14,
                                      0.14,
                                      0.08,
                                      0.08,
                                      0.08,
                                      0.08,
                                      0.08,
                                      0.05,
                                      0.05,
                                      0.05,
                                      0.05,
                                      0.05};

        // Big grains
        center_list.push_back(dealii::Point<dim>(0.2, 0.15));
        center_list.push_back(dealii::Point<dim>(0.25, 0.7));
        center_list.push_back(dealii::Point<dim>(0.5, 0.5));
        center_list.push_back(dealii::Point<dim>(0.6, 0.85));
        center_list.push_back(dealii::Point<dim>(0.85, 0.35));

        // Medium grains
        center_list.push_back(dealii::Point<dim>(0.08, 0.92));
        center_list.push_back(dealii::Point<dim>(0.75, 0.6));
        center_list.push_back(dealii::Point<dim>(0.75, 0.1));
        center_list.push_back(dealii::Point<dim>(0.2, 0.45));
        center_list.push_back(dealii::Point<dim>(0.85, 0.85));

        // Small grains
        center_list.push_back(dealii::Point<dim>(0.55, 0.05));
        center_list.push_back(dealii::Point<dim>(0.1, 0.35));
        center_list.push_back(dealii::Point<dim>(0.95, 0.65));
        center_list.push_back(dealii::Point<dim>(0.9, 0.15));
        center_list.push_back(dealii::Point<dim>(0.45, 0.25));

        scalar_value = 0.0;

        // The big grains
        double dist = 0.0;
        for (unsigned int dir = 0; dir < dim; dir++)
          {
            dist += (point[dir] - center_list[index][dir] * mesh_size[dir]) *
                    (point[dir] - center_list[index][dir] * mesh_size[dir]);
          }
        dist = std::sqrt(dist);
        scalar_value +=
          0.5 * (1.0 - std::tanh((dist - radius[index] * mesh_size[0]) / 0.5));

        // The medium grains
        dist = 0.0;
        for (unsigned int dir = 0; dir < dim; dir++)
          {
            dist += (point[dir] - center_list[index + 5][dir] * mesh_size[dir]) *
                    (point[dir] - center_list[index + 5][dir] * mesh_size[dir]);
          }
        dist = std::sqrt(dist);
        scalar_value +=
          0.5 * (1.0 - std::tanh((dist - radius[index + 5] * mesh_size[0]) / 0.5));

        // The small grains
        dist = 0.0;
        for (unsigned int dir = 0; dir < dim; dir++)
          {
            dist += (point[dir] - center_list[index + 10][dir] * mesh_size[dir]) *
                    (point[dir] - center_list[index + 10][dir] * mesh_size[dir]);
          }
        dist = std::sqrt(dist);
        scalar_value +=
          0.5 * (1.0 - std::tanh((dist - radius[index + 10] * mesh_size[0]) / 0.5));
      }
    else
      {
        scalar_value = 0.0;
      }
  }

  void
  compute_rhs([[maybe_unused]] FieldContainer<dim, degree, number> &variable_list,
              [[maybe_unused]] const SimulationTimer               &sim_timer,
              [[maybe_unused]] unsigned int solve_block_id) const override
  {
    // Instead of checking a hard-coded index value, we can use our custom
    // function to query the current solve block
    if (is_explicit_block(solve_block_id)) // Explicit solve for the order parameters
      {
        ScalarValue f_multiwell;
        ScalarValue ni;
        ScalarValue nj;
        ScalarGrad  nix;

        std::vector<ScalarValue> value_terms(num_grains);
        std::vector<ScalarGrad>  gradient_terms(num_grains);

        // Calculate the evolution equations for the current solve block
        for (unsigned int i = solve_block_start(solve_block_id); i < solve_block_end(solve_block_id); i++)
          {
            ni          = variable_list.template get_value<Scalar, OldOne>(i);
            nix         = variable_list.template get_gradient<Scalar, OldOne>(i);
            f_multiwell = -ni + ni * ni * ni;
            for (unsigned int j = 0; j < num_grains; j++)
              {
                if (i != j)
                  {
                    nj = variable_list.template get_value<Scalar, OldOne>(j);
                    f_multiwell += 2.0 * alpha * ni * nj * nj;
                  }
              }
            value_terms[i] =
              ni - sim_timer.get_timestep() * kinetic_coef * m_well * f_multiwell;
            gradient_terms[i] = -sim_timer.get_timestep() * kinetic_coef * kappa * nix;
          }

        // Submit the RHS terms (but only for the current solve block)
        for (unsigned int i = solve_block_start(solve_block_id); i < solve_block_end(solve_block_id); i++)
          {
            variable_list.set_value_term(i, value_terms[i]);
            variable_list.set_gradient_term(i, gradient_terms[i]);
          }
      }
    else if (is_postprocessing_block(solve_block_id)) // Postprocessing
      {
        ScalarValue f_total = 0.0;
        ScalarValue sum2op  = 0.0;
        ScalarValue ni;
        ScalarValue nj;
        ScalarGrad  nix;

        // Empty region is -1
        ScalarValue max_op_id    = -1.0;
        ScalarValue max_op_value = std::numeric_limits<number>::lowest();

        for (unsigned int i = 0; i < num_grains; i++)
          {
            ni  = variable_list.template get_value<Scalar, Current>(i);
            nix = variable_list.template get_gradient<Scalar, Current>(i);
            f_total += m_well * (0.25 + (ni * ni * ni * ni / 4.0) - (ni * ni / 2.0));
            for (unsigned int j = 0; j < num_grains; j++)
              {
                if (i != j)
                  {
                    nj = variable_list.template get_value<Scalar, Current>(j);
                    f_total += m_well * alpha * ni * ni * nj * nj;
                    sum2op += ni * nj;
                  }
              }
            f_total += (kappa / 2.0) * nix.norm_square();

            for (unsigned int v = 0; v < ScalarValue::size(); v++)
              {
                // We have a lower bound to avoid categorizing the background into grains
                if (ni[v] > max_op_value[v] && ni[v] > 0.05)
                  {
                    max_op_value[v] = ni[v];
                    max_op_id[v]    = i;
                  }
              }
          }

        variable_list.set_value_term(num_grains, sum2op);
        variable_list.set_value_term(num_grains + 1, f_total);
        variable_list.set_value_term(num_grains + 2, max_op_id);
      }
  }

  // These are the grain growth model parameters (see grain_growth.md):
  // m_well is the multi-well coefficient (J / m^3)
  // kappa is the gradient energy coefficient (J / m)
  // alpha is the coefficient of the interaction term in the multi-well potential (unitless)
  // kinetic_coef is the kinetic coefficient of the Allen-Cahn evolution equation (m^4 / (J s)) 
  number m_well;
  number kappa;
  number alpha;
  number kinetic_coef;

  // The number of order parameters and the number of explicit solve blocks
  unsigned int num_grains;
  unsigned int num_explicit_blocks;
  unsigned int postprocess_block_id;

  // These vectors store the field attributes and solve blocks
  std::vector<FieldAttributes> field_attributes;
  std::vector<SolveBlock> solve_blocks;

  // A map of field indices to their solve block indices
  std::map<unsigned int, unsigned int> field_to_block_id;

  // Due to a deal.II bug, we can't have more than 20 fields per solve block.
  // This serves as a workaround for that issue, by dividing the order parameters
  // into multiple solve blocks
  constexpr static unsigned int max_fields_per_solve_block = 20;
};

PRISMS_PF_END_NAMESPACE
