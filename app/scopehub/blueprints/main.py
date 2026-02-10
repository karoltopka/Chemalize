"""
ScopeHub main blueprint - Scientific search query generator routes
"""
from flask import Blueprint, render_template, session, g, request, jsonify, send_file
from werkzeug.utils import secure_filename
from app.nocache import nocache
from app.scopehub.utils.excel_parser import (
    parse_excel_file, save_to_excel, get_sections, get_subsections,
    add_entry, update_entry, get_entries_by_section, create_empty_sheets,
    sheets_to_unified, unified_to_sheets
)
from app.scopehub.utils.llm_generator import (
    generate_search_query, refine_search_query, combine_queries, generate_all_queries
)
from app.config import get_user_id
import os
import tempfile

scopehub_main_bp = Blueprint('scopehub_main', __name__)

# Store data in session (temporary - later move to database)
def get_scopehub_data():
    """Get ScopeHub data from session"""
    user_id = get_user_id()
    session_key = f'scopehub_data_{user_id}'

    if session_key not in session:
        session[session_key] = create_empty_sheets()

    return session[session_key]


def set_scopehub_data(data):
    """Save ScopeHub data to session"""
    user_id = get_user_id()
    session_key = f'scopehub_data_{user_id}'
    session[session_key] = data
    session.modified = True


def get_scopehub_data_unified():
    """Get ScopeHub data in unified format (one list with all 3 formulas)"""
    sheets = get_scopehub_data()
    unified = sheets_to_unified(sheets)
    return unified


def set_scopehub_data_unified(unified_data):
    """Save unified data back to session (converts to 3 sheets)"""
    sheets = unified_to_sheets(unified_data)
    set_scopehub_data(sheets)


@scopehub_main_bp.route('/')
@nocache
def scopehub_home():
    """ScopeHub home page - scientific search query generator dashboard"""
    return render_template('scopehub/home.html')


@scopehub_main_bp.route('/database')
@nocache
def scopehub_database():
    """ScopeHub query database - manage search queries and templates"""
    data = get_scopehub_data()
    return render_template('scopehub/database.html', data=data)


@scopehub_main_bp.route('/query-manager')
@nocache
def query_manager():
    """ScopeHub query manager - interactive interface for query management"""
    return render_template('scopehub/query_manager.html')


@scopehub_main_bp.route('/upload', methods=['POST'])
@nocache
def upload_excel():
    """Upload keyword_strings.xlsx file"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if file and file.filename.endswith(('.xlsx', '.xls')):
        # Save temporarily
        temp_path = os.path.join(tempfile.gettempdir(), secure_filename(file.filename))
        file.save(temp_path)

        # Parse file - returns (data, is_valid_format, warning_message)
        data, is_valid_format, warning_message = parse_excel_file(temp_path)

        # Save to session
        set_scopehub_data(data)

        # Clean up
        os.remove(temp_path)

        response_data = {
            'success': True,
            'sheets': list(data.keys())
        }

        # Add warning if file format is not proper
        if not is_valid_format and warning_message:
            response_data['warning'] = warning_message
            response_data['message'] = 'File uploaded with warnings'
        else:
            response_data['message'] = 'File uploaded successfully'

        return jsonify(response_data)

    return jsonify({'error': 'Invalid file format'}), 400


@scopehub_main_bp.route('/api/data/<sheet_name>', methods=['GET'])
@nocache
def get_sheet_data(sheet_name):
    """Get data for specific sheet"""
    data = get_scopehub_data()

    if sheet_name not in data:
        return jsonify({'error': 'Sheet not found'}), 404

    df = data[sheet_name]

    # Convert to JSON-serializable format
    records = df.to_dict('records')

    return jsonify({
        'sheet': sheet_name,
        'data': records,
        'sections': get_sections(df)
    })


@scopehub_main_bp.route('/api/generate', methods=['POST'])
@nocache
def generate_query():
    """Generate search query using LLM"""
    req_data = request.get_json()

    name = req_data.get('name', '')
    database = req_data.get('database', 'Pubmed')
    context = req_data.get('context', {})

    if not name:
        return jsonify({'error': 'Name is required'}), 400

    # Generate query
    result = generate_search_query(name, database, context)

    return jsonify({
        'success': True,
        'searching_formula': result['query'],
        'raw_output': result['raw'],
        'has_error': result['has_error']
    })


@scopehub_main_bp.route('/api/generate_batch', methods=['POST'])
@nocache
def generate_batch():
    """Generate queries for multiple entries at once"""
    req_data = request.get_json()

    sheet_name = req_data.get('sheet', 'Pubmed')
    entries = req_data.get('entries', [])

    if not entries:
        return jsonify({'error': 'No entries to generate'}), 400

    data = get_scopehub_data()

    if sheet_name not in data:
        return jsonify({'error': 'Invalid sheet'}), 400

    generated_count = 0

    # Generate for each entry
    for entry in entries:
        index = entry.get('index')
        name = entry.get('name', '')
        section = entry.get('section', '')
        subsection = entry.get('subsection', '')

        if index is not None and name:
            # Generate query
            result = generate_search_query(
                name,
                sheet_name,
                {'section': section, 'subsection': subsection}
            )

            # Update entry - formula
            data[sheet_name] = update_entry(
                data[sheet_name],
                index,
                'Searching Formula',
                result['query']
            )

            # Update entry - raw output
            data[sheet_name] = update_entry(
                data[sheet_name],
                index,
                'Raw_Output',
                result['raw']
            )

            generated_count += 1

    # Save updated data
    set_scopehub_data(data)

    return jsonify({
        'success': True,
        'generated': generated_count,
        'message': f'Generated {generated_count} queries'
    })


@scopehub_main_bp.route('/api/add_entry', methods=['POST'])
@nocache
def add_new_entry():
    """Add new entry to database"""
    req_data = request.get_json()

    sheet_name = req_data.get('sheet', 'Pubmed')
    section = req_data.get('section', '')
    subsection = req_data.get('subsection', '')
    name = req_data.get('name', '')
    searching_formula = req_data.get('searching_formula', '')
    string_combined = req_data.get('string_combined', '')
    comments = req_data.get('comments', '')

    # Only section and subsection are required, name can be empty (user fills later)
    if not section or not subsection:
        return jsonify({'error': 'Section and subsection are required'}), 400

    data = get_scopehub_data()

    if sheet_name not in data:
        return jsonify({'error': 'Invalid sheet'}), 400

    # Add entry
    data[sheet_name] = add_entry(
        data[sheet_name], section, subsection, name,
        searching_formula, string_combined, comments
    )

    set_scopehub_data(data)

    return jsonify({'success': True, 'message': 'Entry added successfully'})


@scopehub_main_bp.route('/api/update_entry', methods=['POST'])
@nocache
def update_existing_entry():
    """Update existing entry"""
    req_data = request.get_json()

    sheet_name = req_data.get('sheet', 'Pubmed')
    index = req_data.get('index')
    column = req_data.get('column')
    value = req_data.get('value', '')

    if index is None or not column:
        return jsonify({'error': 'Index and column are required'}), 400

    data = get_scopehub_data()

    if sheet_name not in data:
        return jsonify({'error': 'Invalid sheet'}), 400

    # Update entry
    data[sheet_name] = update_entry(data[sheet_name], index, column, value)

    set_scopehub_data(data)

    return jsonify({'success': True, 'message': 'Entry updated successfully'})


@scopehub_main_bp.route('/api/delete_section', methods=['POST'])
@nocache
def delete_section():
    """Delete all entries in a section"""
    req_data = request.get_json()

    sheet_name = req_data.get('sheet', 'Pubmed')
    section = req_data.get('section')

    if not section:
        return jsonify({'error': 'Section is required'}), 400

    data = get_scopehub_data()

    if sheet_name not in data:
        return jsonify({'error': 'Invalid sheet'}), 400

    # Filter out all entries with the specified section
    df = data[sheet_name]
    initial_count = len(df)
    data[sheet_name] = df[df['Section'] != section]
    deleted_count = initial_count - len(data[sheet_name])

    set_scopehub_data(data)

    return jsonify({
        'success': True,
        'message': f'Section deleted successfully',
        'deleted_entries': deleted_count
    })


@scopehub_main_bp.route('/export', methods=['GET'])
@nocache
def export_excel():
    """Export data back to Excel file"""
    data = get_scopehub_data()

    # Create temporary file
    temp_path = os.path.join(tempfile.gettempdir(), 'keyword_strings_export.xlsx')

    # Save to Excel
    if save_to_excel(data, temp_path):
        return send_file(
            temp_path,
            as_attachment=True,
            download_name='keyword_strings.xlsx',
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )

    return jsonify({'error': 'Failed to export file'}), 500


# ============================================================================
# UNIFIED API ENDPOINTS - One topic list with 3 database formulas
# ============================================================================

@scopehub_main_bp.route('/api/data_unified', methods=['GET'])
@nocache
def get_data_unified():
    """Get unified data (one list with formulas for all 3 databases)"""
    try:
        data = get_scopehub_data_unified()

        # Clean data to ensure JSON serialization works
        # Replace NaN, None, and other problematic values with empty strings
        cleaned_data = []
        for entry in data:
            cleaned_entry = {}
            for key, value in entry.items():
                # Handle pandas NaN, None, and other non-serializable values
                if value is None or (isinstance(value, float) and str(value) == 'nan'):
                    cleaned_entry[key] = ''
                else:
                    cleaned_entry[key] = str(value) if not isinstance(value, (str, int, float, bool)) else value
            cleaned_data.append(cleaned_entry)

        return jsonify({'success': True, 'data': cleaned_data})
    except Exception as e:
        import traceback
        print(f"ERROR in get_data_unified: {e}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


@scopehub_main_bp.route('/api/update_entry_unified', methods=['POST'])
@nocache
def update_entry_unified():
    """Update a field in unified data"""
    req_data = request.get_json()

    index = req_data.get('index')
    column = req_data.get('column')
    value = req_data.get('value', '')

    if index is None or not column:
        return jsonify({'error': 'Index and column are required'}), 400

    try:
        data = get_scopehub_data_unified()

        if index >= len(data):
            return jsonify({'error': 'Invalid index'}), 400

        # Update the field
        data[index][column] = value

        # Save back
        set_scopehub_data_unified(data)

        return jsonify({'success': True, 'message': 'Entry updated successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@scopehub_main_bp.route('/api/add_entry_unified', methods=['POST'])
@nocache
def add_entry_unified():
    """Add new entry to unified data"""
    req_data = request.get_json()

    section = req_data.get('section', '').strip()
    subsection = req_data.get('subsection', '').strip()
    name = req_data.get('name', '')
    comments = req_data.get('comments', '')


    if not section:
        return jsonify({'error': 'Section is required'}), 400

    try:
        data = get_scopehub_data_unified()

        # If no subsection provided, auto-generate from existing entries
        if not subsection:
            # Count entries in this section
            section_entries = [e for e in data if e.get('Section') == section]
            if section_entries:
                # Find max subsection number
                max_num = 0
                for entry in section_entries:
                    sub = entry.get('Subsection', '')
                    if sub:
                        # Extract number from subsection like "A1" -> 1
                        match = sub.replace(section, '').strip()
                        try:
                            num = int(match) if match else 0
                            max_num = max(max_num, num)
                        except ValueError:
                            pass
                subsection = f"{section}{max_num + 1}"
            else:
                # First entry in section - use section + 1
                subsection = f"{section}1"


        # Add new entry
        new_entry = {
            'Section': section,
            'Subsection': subsection,
            'Name': name,
            'Name_Corrected': '',
            'Pubmed_Formula': '',
            'WOS_Formula': '',
            'Scopus_Formula': '',
            'Comments': comments,
            'Pubmed_Raw': '',
            'WOS_Raw': '',
            'Scopus_Raw': ''
        }

        data.append(new_entry)

        set_scopehub_data_unified(data)

        return jsonify({'success': True, 'message': 'Entry added successfully'})
    except Exception as e:
        import traceback
        print(f"ERROR in add_entry_unified: {e}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


@scopehub_main_bp.route('/api/delete_entry_unified', methods=['POST'])
@nocache
def delete_entry_unified():
    """Delete a single entry from unified data"""
    req_data = request.get_json()

    index = req_data.get('index')

    if index is None:
        return jsonify({'error': 'Index is required'}), 400

    try:
        data = get_scopehub_data_unified()

        if index >= len(data) or index < 0:
            return jsonify({'error': 'Invalid index'}), 400

        # Remove entry at index
        data.pop(index)

        set_scopehub_data_unified(data)

        return jsonify({
            'success': True,
            'message': 'Entry deleted successfully'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@scopehub_main_bp.route('/api/delete_section_unified', methods=['POST'])
@nocache
def delete_section_unified():
    """Delete all entries in a section from unified data"""
    req_data = request.get_json()

    section = req_data.get('section')

    if not section:
        return jsonify({'error': 'Section is required'}), 400

    try:
        data = get_scopehub_data_unified()
        initial_count = len(data)

        # Filter out entries with specified section
        data = [entry for entry in data if entry.get('Section') != section]

        deleted_count = initial_count - len(data)

        set_scopehub_data_unified(data)

        return jsonify({
            'success': True,
            'message': 'Section deleted successfully',
            'deleted_entries': deleted_count
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@scopehub_main_bp.route('/api/generate_batch_unified', methods=['POST'])
@nocache
def generate_batch_unified():
    """Generate queries for multiple entries at once - all 3 databases"""
    req_data = request.get_json()

    entries_to_generate = req_data.get('entries', [])
    force_regenerate = req_data.get('force_regenerate', False)  # Force regeneration even if formulas exist

    if not entries_to_generate:
        return jsonify({'error': 'No entries provided'}), 400

    try:
        data = get_scopehub_data_unified()
        generated_count = 0

        for entry_info in entries_to_generate:
            index = entry_info.get('index')
            name = entry_info.get('name')
            section = entry_info.get('section')
            subsection = entry_info.get('subsection')

            if index is not None and name and index < len(data):
                current_entry = data[index]

                # Check which formulas are missing
                needs_pubmed = not current_entry.get('Pubmed_Formula', '').strip()
                needs_wos = not current_entry.get('WOS_Formula', '').strip()
                needs_scopus = not current_entry.get('Scopus_Formula', '').strip()

                # Generate if formulas are missing OR if force_regenerate is true
                should_generate = force_regenerate or needs_pubmed or needs_wos or needs_scopus

                if should_generate:
                    # Generate all 3 queries at once (PubMed via LLM, WOS/Scopus via regex)
                    result = generate_all_queries(
                        name,
                        {'section': section, 'subsection': subsection}
                    )

                    corrected_names = {'wos': None, 'scopus': None, 'pubmed': None}
                    errors = []

                    # Store PubMed results
                    data[index]['Pubmed_Formula'] = result.get('Pubmed_Formula', '')
                    data[index]['Pubmed_Raw'] = result.get('Pubmed_Raw', '')

                    # Store WOS results
                    data[index]['WOS_Formula'] = result.get('WOS_Formula', '')
                    data[index]['WOS_Raw'] = result.get('WOS_Raw', '')

                    # Store Scopus results
                    data[index]['Scopus_Formula'] = result.get('Scopus_Formula', '')
                    data[index]['Scopus_Raw'] = result.get('Scopus_Raw', '')

                    # Extract corrected names from formulas
                    from app.scopehub.utils.llm_generator import extract_corrected_name

                    if result.get('Pubmed_Formula'):
                        corrected = extract_corrected_name(result['Pubmed_Formula'], 'Pubmed')
                        if corrected:
                            corrected_names['pubmed'] = corrected

                    if result.get('WOS_Formula'):
                        corrected = extract_corrected_name(result['WOS_Formula'], 'WOS')
                        if corrected:
                            corrected_names['wos'] = corrected

                    if result.get('Scopus_Formula'):
                        corrected = extract_corrected_name(result['Scopus_Formula'], 'Scopus')
                        if corrected:
                            corrected_names['scopus'] = corrected

                    # Check for errors (empty formulas indicate failure)
                    if not result.get('Pubmed_Formula'):
                        errors.append("PubMed: Failed to generate query")
                    if not result.get('WOS_Formula'):
                        errors.append("WOS: Failed to generate query")
                    if not result.get('Scopus_Formula'):
                        errors.append("Scopus: Failed to generate query")

                    # Store errors in Comments field if any
                    if errors:
                        data[index]['Comments'] = '\n'.join(errors)
                        print(f"ERRORS for '{name}': {'; '.join(errors)}")

                    # Set Name_Corrected - prioritize WOS > Scopus > PubMed
                    # WOS has simplest formulas, most reliable for name extraction
                    # Only set if it's a spelling correction, not a concept change
                    best_corrected = None
                    if corrected_names['wos']:
                        best_corrected = corrected_names['wos']
                    elif corrected_names['scopus']:
                        best_corrected = corrected_names['scopus']
                    elif corrected_names['pubmed']:
                        best_corrected = corrected_names['pubmed']

                    if best_corrected:
                        from app.scopehub.utils.llm_generator import is_spelling_correction
                        if is_spelling_correction(name, best_corrected):
                            data[index]['Name_Corrected'] = best_corrected
                        else:
                            pass

                    generated_count += 1

        set_scopehub_data_unified(data)

        return jsonify({
            'success': True,
            'generated': generated_count,
            'message': f'Generated {generated_count} queries for all databases'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@scopehub_main_bp.route('/settings')
@nocache
def settings():
    """ScopeHub settings page - configure LLM provider"""
    from app.scopehub.utils import llm_generator

    # Get current settings
    current_backend = llm_generator.LLM_BACKEND

    # Get available backends
    backends = {
        'lm_studio': {
            'name': 'LM Studio',
            'description': 'Local LLM server (OpenAI-compatible)',
            'available': True
        },
        'openwebui': {
            'name': 'OpenWebUI',
            'description': 'Open WebUI API with Ollama support',
            'available': True
        },
        'google_ai_studio': {
            'name': 'Google AI Studio',
            'description': 'Google Gemini API (cloud-based)',
            'available': llm_generator.GOOGLE_AI_AVAILABLE and bool(llm_generator.GOOGLE_AI_STUDIO_API_KEY)
        }
    }

    # Get provider-specific settings
    settings_data = {
        'current_backend': current_backend,
        'backends': backends,
        'lm_studio': {
            'url': llm_generator.LM_STUDIO_URL,
            'timeout': llm_generator.LM_STUDIO_TIMEOUT
        },
        'openwebui': {
            'url': llm_generator.OPENWEBUI_URL,
            'model': llm_generator.OPENWEBUI_MODEL,
            'timeout': llm_generator.OPENWEBUI_TIMEOUT,
            'has_api_key': bool(llm_generator.OPENWEBUI_API_KEY)
        },
        'google_ai_studio': {
            'model': llm_generator.GOOGLE_AI_STUDIO_MODEL,
            'timeout': llm_generator.GOOGLE_AI_STUDIO_TIMEOUT,
            'has_api_key': bool(llm_generator.GOOGLE_AI_STUDIO_API_KEY),
            'sdk_installed': llm_generator.GOOGLE_AI_AVAILABLE
        }
    }

    return render_template('scopehub/settings.html', settings=settings_data)


@scopehub_main_bp.route('/api/update_llm_backend', methods=['POST'])
@nocache
def update_llm_backend():
    """Update LLM backend selection (session-based, requires app restart for env vars)"""
    req_data = request.get_json()
    backend = req_data.get('backend', 'lm_studio')

    # Validate backend
    valid_backends = ['lm_studio', 'openwebui', 'google_ai_studio']
    if backend not in valid_backends:
        return jsonify({'error': 'Invalid backend'}), 400

    # Store in session (temporary until app restart)
    session['llm_backend_override'] = backend

    return jsonify({
        'success': True,
        'backend': backend,
        'message': f'Backend switched to {backend}. Note: This is temporary. To make permanent, set LLM_BACKEND={backend} in your .env file and restart the application.'
    })
