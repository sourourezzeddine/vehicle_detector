# Functions
def get_country_from_text(text):
    """
    Get the country name and the number of digits in the national identification number from the provided text.

    This function checks the provided text against predefined patterns, keywords, or other identifiers 
    for each country's national identification number. It returns the country name and the number of digits 
    in the national identification number if a match is found.

    :param text: The text to be analyzed.
    :type text: str
    :return: A tuple containing the country name and the number of digits in the national identification number.
             If no match is found, it returns 'Unknown' and None.
    :rtype: tuple[str, int] or tuple[str, None]
    """
    # Define patterns, keywords, or other identifiers for each country along with the number of digits in the national identification number
    country_patterns = {
        'Austria': ['Austria', 'Österreich', 'AT', 'AUT', 9],
        'Belgium': ['Belgium', 'Belgique', 'België', 'Belgien', 'BE', 'BEL', 11],
        'Bulgaria': ['Bulgaria', 'България', 'BG', 'BGR', 10],
        'Croatia': ['Croatia', 'Hrvatska', 'HR', 'HRV', 11],
        'Cyprus': ['Cyprus', 'Κύπρος', 'Kıbrıs', 'CY', 'CYP', 9],
        'Czech Republic': ['Czech Republic', 'Česká republika', 'CZ', 'CZE', 10],
        'Estonia': ['Estonia', 'Eesti', 'EE', 'EST', 11],
        'Finland': ['Finland', 'Suomi', 'FI', 'FIN', 11],
        'France': ['France', 'FR', 'FRA', 'F','AA','AA' , 'AB' ,'AC' , 'AD' , 'AE' ,'AF', 'AG' , 'AH' ,'AI', 'AJ' ,'AK' ,'AL' ,'AM' ,'AN', 'AO', 'AP' ,'AQ', 'AR' ,'AS', 'AT', 'AU' ,'AV', 'AW' ,'AX', 'AY', 'AZ','BA' ,'BB' ,'BC', 'BD', 'BE' ,'BF','BG', 'BH', 'BI', 'BJ','BK', 'BL', 'BM','BN','BO','BP','BQ','BR','BS','BT','BU', 'BV','BW','BX' ,'BY','BZ', 15],
        'Germany': ['Germany', 'Deutschland', 'DE', 'DEU','BD','BP','BW','THW' ,11],
        'Greece': ['Greece', 'Ελλάδα', 'GR', 'GRC', 9],
        'Hungary': ['Hungary', 'Magyarország', 'HU', 'HUN', 11],
        'Ireland': ['Ireland', 'Éire', 'IE', 'IRL', 9],
        'Italy': ['Italy', 'Italia', 'IT', 'ITA', 16],
        'Lithuania': ['Lithuania', 'Lietuva', 'LT', 'LTU', 11],
        'Luxembourg': ['Luxembourg', 'Luxemburg', 'Lëtzebuerg', 'LU', 'LUX', 13],
        'Netherlands': ['Netherlands', 'Nederland', 'NL', 'NLD', 9],      
        'Portugal': ['Portugal', 'PT', 'PRT', 9],
        'Romania': ['Romania', 'România', 'RO', 'ROU', 13],
        'Slovakia': ['Slovakia', 'Slovensko', 'SK', 'SVK', 10],
        'Spain': ['Spain', 'España', 'ES', 'ESP','ALB', 'AB', 'AL', 9],
        'Sweden': ['Sweden', 'Sverige', 'SE', 'SWE', 12],
        'Tunisia': ['Tunisia', 'تونس', 'TN', 'TUN', 8],
        'United States': ['United States', 'USA', 'America','NEW YORK', 'NEW' , 'US', 'United States of America', 'US', None], 
        'United Kingdom': ['GB','United Kingdom', 'UK', 'Great Britain', 'GB', 'England', 'Scotland', 'Wales', 'Northern Ireland', 'GBR', 'UK','A','B','C', None],
        # Add other countries as needed
    }
    
    # Iterate through each country and check for matches
    for country, keywords in country_patterns.items():
        for keyword in keywords[:-1]: # excluding the last item which is the number of digits
            if keyword.lower() in text.lower():
                return country, keywords[-1] # return the country and the number of digits
    
    # If no country is matched, return 'Unknown' and None for the number of digits
    return 'Unknown', None

