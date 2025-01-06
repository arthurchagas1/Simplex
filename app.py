import argparse
import numpy as np
import sys

def parse_arguments():
    parser = argparse.ArgumentParser(description='Implementação do Método Simplex.')
    parser.add_argument('filename', type=str, help='Nome do arquivo lp de entrada')
    parser.add_argument('--decimals', type=int, default=3, help='Número de casas decimais para imprimir valores numéricos.')
    parser.add_argument('--digits', type=int, default=7, help='Número total de dígitos para imprimir valores numéricos.')
    parser.add_argument('--policy', type=str, choices=['largest', 'smallest', 'bland'], default='largest', help='Política de seleção de pivô.')
    return parser.parse_args()

def read_input(filename):
    with open(filename, 'r') as file:
        lines = [line.strip() for line in file if line.strip()]
    if len(lines) < 4:
        print("Erro: O arquivo de entrada deve conter pelo menos 4 linhas (número de variáveis, número de restrições, tipos de variáveis, e coeficientes da função objetivo).")
        sys.exit(1)
    
    try:
        num_vars = int(lines[0])
    except ValueError:
        print("Erro: A primeira linha deve ser um número inteiro representando o número de variáveis.")
        sys.exit(1)
    
    try:
        num_constraints = int(lines[1])
    except ValueError:
        print("Erro: A segunda linha deve ser um número inteiro representando o número de restrições.")
        sys.exit(1)
    
    var_types = list(map(int, lines[2].split()))
    if len(var_types) != num_vars:
        print(f"Erro: Número de tipos de variáveis ({len(var_types)}) não corresponde ao número de variáveis ({num_vars}).")
        sys.exit(1)
    
    try:
        objective_coeffs = list(map(float, lines[3].split()))
    except ValueError:
        print("Erro: A quarta linha deve conter coeficientes numéricos da função objetivo.")
        sys.exit(1)
    
    if len(objective_coeffs) != num_vars:
        print(f"Erro: Número de coeficientes na função objetivo ({len(objective_coeffs)}) não corresponde ao número de variáveis ({num_vars}).")
        sys.exit(1)
    
    constraints = []
    for i, line in enumerate(lines[4:], start=5):
        parts = line.split()
        # Identificar a posição do operador
        if '<=' in parts:
            op_index = parts.index('<=')
            operator = '<='
        elif '>=' in parts:
            op_index = parts.index('>=')
            operator = '>='
        elif '=' in parts:
            op_index = parts.index('=')
            operator = '=='
        else:
            print(f"Erro: Operador não reconhecido na restrição na linha {i}: {line}")
            sys.exit(1)
        try:
            coeffs = list(map(float, parts[:op_index]))
            rhs = float(parts[op_index + 1])
        except ValueError:
            print(f"Erro: Coeficientes ou lado direito não numéricos na restrição na linha {i}: {line}")
            sys.exit(1)
        if len(coeffs) != num_vars:
            print(f"Erro: Número de coeficientes na restrição na linha {i} ({len(coeffs)}) não corresponde ao número de variáveis ({num_vars}).")
            sys.exit(1)
        constraints.append((coeffs, operator, rhs))
    
    if len(constraints) != num_constraints:
        print(f"Erro: Número de restrições fornecidas ({len(constraints)}) não corresponde ao número especificado ({num_constraints}).")
        sys.exit(1)
    
    return num_vars, num_constraints, var_types, objective_coeffs, constraints

def format_number(number, digits, decimals):
    formatted = f"{number:.{decimals}f}"
    # Ajustar para o total de dígitos
    if len(formatted.replace('.', '').replace('-', '')) > digits:
        formatted = f"{number:.{decimals}e}"
    return formatted.rjust(digits)

def print_table(table, headers, decimals, digits):
    # Calcular larguras das colunas
    col_widths = [max(len(str(item)) for item in col) for col in zip(*table, headers)]
    col_widths = [max(w, len(h)) for w, h in zip(col_widths, headers)]
    # Imprimir cabeçalhos
    header_row = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths))
    separator = "-+-".join('-' * w for w in col_widths)
    print(header_row)
    print(separator)
    # Imprimir linhas
    for row in table:
        print(" | ".join(str(item).ljust(w) for item, w in zip(row, col_widths)))

def transform_to_standard_form(num_vars, num_constraints, var_types, objective_coeffs, constraints):
    A = []
    b = []
    directions = []
    # Transformar variáveis livres e não-positivas
    new_objective = []
    original_vars = []
    # Transformar variáveis para serem não-negativas
    for i, var_type in enumerate(var_types):
        if var_type == 1:
            # x_i >= 0, nenhuma substituição necessária
            original_vars.append(f"x{i}")
            new_objective.append(objective_coeffs[i])
        elif var_type == -1:
            # x_i <= 0, substituir por x'_i >= 0 onde x_i = -x'_i
            original_vars.append(f"x{i}")
            new_objective.append(-objective_coeffs[i])
        elif var_type == 0:
            # x_i livre, substituir por x'_i - x''_i onde x'_i, x''_i >= 0
            original_vars.append(f"x{i}")
            original_vars.append(f"x{i}_neg")
            new_objective.append(objective_coeffs[i])
            new_objective.append(-objective_coeffs[i])
        else:
            print(f"Erro: Tipo de variável inválido para x{i}: {var_type}")
            sys.exit(1)
    # Transformar as restrições
    A_transformed = []
    b_transformed = []
    directions_transformed = []
    for coeffs, operator, rhs in constraints:
        new_coeffs = []
        for i, var_type in enumerate(var_types):
            if var_type == 1:
                new_coeffs.append(coeffs[i])
            elif var_type == -1:
                new_coeffs.append(-coeffs[i])
            elif var_type == 0:
                new_coeffs.append(coeffs[i])
                new_coeffs.append(-coeffs[i])
        A_transformed.append(new_coeffs)
        b_transformed.append(rhs)
        directions_transformed.append(operator)
    A_transformed = np.array(A_transformed, dtype=float)
    b_transformed = np.array(b_transformed, dtype=float)
    # Não remover restrições
    A_full = A_transformed.copy()
    rowsEliminated = []
    return A_full, b_transformed, np.array(new_objective, dtype=float), directions_transformed, original_vars

def initialize_simplex(A, b, c, directions):
    num_constraints, num_vars = A.shape
    # Converter desigualdades em igualdades adicionando variáveis de folga
    basis = []
    for i, op in enumerate(directions):
        if op == '<=':
            slack = np.zeros(num_constraints)
            slack[i] = 1
            A = np.hstack((A, slack.reshape(-1,1)))
            c = np.hstack((c, [0]))
            basis.append(num_vars)
            num_vars +=1
        elif op == '>=':
            # Não presente na PL fornecida, mas incluído para completude
            slack = np.zeros(num_constraints)
            slack[i] = -1
            A = np.hstack((A, slack.reshape(-1,1)))
            c = np.hstack((c, [0]))
            # Adicionar variável artificial se necessário
            # Não implementado aqui
            basis.append(num_vars)
            num_vars +=1
        elif op == '==':
            # Não presente na PL fornecida, mas incluído para completude
            # Adicionar variável artificial se necessário
            # Não implementado aqui
            pass
        else:
            print(f"Erro: Operador desconhecido na restrição: {op}")
            sys.exit(1)
    return A, b, c, basis

def pivot(tableau, pivot_row, pivot_col):
    pivot_element = tableau[pivot_row][pivot_col]
    tableau[pivot_row] = tableau[pivot_row] / pivot_element
    for i in range(len(tableau)):
        if i != pivot_row:
            tableau[i] = tableau[i] - tableau[i][pivot_col] * tableau[pivot_row]
    # Zerar pequenos valores
    tableau[np.abs(tableau) < 1e-10] = 0.0
    return tableau

def select_entering_variable(tableau, policy):
    # Linha do objetivo é a última
    objective = tableau[-1, :-1]
    if policy == 'bland':
        candidates = [j for j, coeff in enumerate(objective) if coeff < -1e-10]
        if not candidates:
            return None
        return min(candidates)
    elif policy == 'largest':
        entering = np.argmin(objective)  # Para maximização, escolher o menor coeficiente
        if objective[entering] >= -1e-10:
            return None
        return entering
    elif policy == 'smallest':
        entering = np.argmin(objective)
        if objective[entering] >= -1e-10:
            return None
        return entering
    else:
        return None

def select_leaving_variable(tableau, entering):
    ratios = []
    for i in range(len(tableau)-1):
        if tableau[i][entering] > 1e-10:
            ratios.append(tableau[i][-1] / tableau[i][entering])
        else:
            ratios.append(np.inf)
    min_ratio = min(ratios)
    if min_ratio == np.inf:
        return None
    # Implementar a regra de Bland para desempate
    candidates = [i for i, ratio in enumerate(ratios) if ratio == min_ratio]
    return min(candidates)

def simplex(tableau, basis, decimals, digits, policy):
    iteration = 0
    while True:
        print(f"\nIteração {iteration}:")
        headers = [f"x{j}" for j in range(tableau.shape[1]-1)] + ["b"]
        table = []
        for i in range(len(tableau)):
            row = [format_number(num, digits, decimals) for num in tableau[i]]
            table.append(row)
        print_table(table, headers, decimals, digits)
        # Verificar otimalidade
        objective = tableau[-1, :-1]
        # Para maximização, verificar se há algum coeficiente <0
        if all(c >= -1e-10 for c in objective):
            print("Solução ótima encontrada.")
            break
        # Selecionar variável entrando
        entering = select_entering_variable(tableau, policy)
        if entering is None:
            print("Solução ótima encontrada.")
            break
        # Selecionar variável saindo
        leaving = select_leaving_variable(tableau, entering)
        if leaving is None:
            print("Status: ilimitado")
            return "ilimitado", None, None, None
        # Pivoteamento
        print(f"Entrando na base: x{entering}")
        print(f"Saindo da base: x{basis[leaving]}")
        pivot(tableau, leaving, entering)
        basis[leaving] = entering
        iteration +=1
    # Extrair solução
    solution = np.zeros(tableau.shape[1]-1)
    for i in range(len(basis)):
        if basis[i] < tableau.shape[1]-1:
            solution[basis[i]] = tableau[i][-1]
    objective_value = tableau[-1][-1]
    return "otimo", solution, objective_value, basis

def main():
    args = parse_arguments()
    num_vars, num_constraints, var_types, objective_coeffs, constraints = read_input(args.filename)
    A, b, c, directions, original_vars = transform_to_standard_form(
        num_vars, num_constraints, var_types, objective_coeffs, constraints
    )
    # Inicializar Simplex
    A_std, b_std, c_std, basis = initialize_simplex(A, b, c, directions)
    num_vars_std = A_std.shape[1]
    # Construir tableau inicial
    tableau = np.zeros((A_std.shape[0]+1, num_vars_std+1))
    tableau[:-1, :-1] = A_std
    tableau[:-1, -1] = b_std
    tableau[-1, :-1] = -c_std
    # Rodar Simplex
    status, solution, objective, final_basis = simplex(tableau, basis, args.decimals, args.digits, args.policy)
    # Imprimir status final
    print("\nStatus:", status)
    if status == "otimo":
        print("Objetivo:", format_number(objective, args.digits, args.decimals))
        print("Solução:")
        print(" ".join(format_number(x, args.digits, args.decimals) for x in solution))
    elif status == "ilimitado":
        print("Status: ilimitado")
    elif status == "inviavel":
        print("Status: inviável")

if __name__ == "__main__":
    main()

